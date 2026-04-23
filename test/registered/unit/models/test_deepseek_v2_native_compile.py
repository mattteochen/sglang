import copy
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.layers.attention.nsa_backend import _get_cached_mla_kv_buffer_view
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepseekV2DecoderLayer,
    DeepseekV2MLP,
    DeepseekV2Model,
    _ExperimentalPrefillCompileLayerGroup,
    _deepseek_prefill_mlp_native_bf16_gemms_enabled,
    _mark_experimental_prefill_dynamic_inputs,
    _validate_native_compile_linear_semantics,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, suite="stage-b-test-1-gpu-small")


class FakeNativeCompileLinear(nn.Module):
    def __init__(
        self,
        *,
        gather_output: bool = False,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.gather_output = gather_output
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.bias = nn.Parameter(torch.zeros(1)) if bias else None


class FakePPGroup:
    rank_in_group = 0
    world_size = 1
    is_first_rank = True
    is_last_rank = True


class FakeLayerCommunicator:
    def should_fuse_mlp_allreduce_with_next_layer(self, forward_batch):
        return False

    def should_use_reduce_scatter(self, forward_batch):
        return False


class FakeCompileModeOp(MultiPlatformOp):
    def forward_native(self, x):
        return x

    def forward_cuda(self, x):
        return x

    def forward_cpu(self, x):
        return x


class FakeCompileModeTopK(MultiPlatformOp):
    def forward_native(self, x):
        return x

    def forward_cuda(self, x):
        return x


class FakeCompileModeFusedMoE(MultiPlatformOp):
    def forward_native(self, layer, dispatch_output):
        return dispatch_output

    def forward_cuda(self, layer, dispatch_output):
        return dispatch_output


class FakePrefillCompileLayer(nn.Module):
    def __init__(self, hidden_size: int, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.layer_communicator = FakeLayerCommunicator()
        self.lin1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.compile_mode_op = FakeCompileModeOp()
        self.lin2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_scale = nn.Parameter(torch.randn(hidden_size))

    def _core(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        pos = positions.to(hidden_states.dtype).unsqueeze(-1) * self.pos_scale
        hidden_states = self.compile_mode_op(self.lin1(hidden_states) + pos)
        return self.lin2(torch.relu(hidden_states))

    def _prepare_experimental_prefill_compile_state(self, forward_batch):
        del forward_batch

    def _forward_prefill_impl(
        self,
        positions,
        hidden_states,
        forward_batch,
        residual,
        zero_allocator,
        gemm_output_zero_allocator,
        llama_4_scaling,
        prev_topk_indices=None,
        should_allreduce_fusion=False,
        use_reduce_scatter=False,
    ):
        del (
            forward_batch,
            zero_allocator,
            gemm_output_zero_allocator,
            llama_4_scaling,
            prev_topk_indices,
            should_allreduce_fusion,
            use_reduce_scatter,
        )
        return self._core(positions, hidden_states), residual, None

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        residual,
        zero_allocator,
        gemm_output_zero_allocator,
        llama_4_scaling,
        prev_topk_indices=None,
    ):
        del (
            forward_batch,
            zero_allocator,
            gemm_output_zero_allocator,
            llama_4_scaling,
            prev_topk_indices,
        )
        return self._core(positions, hidden_states), residual, None


class FakeIndexer:
    def __init__(self, supported: bool):
        self.supported = supported
        self.call_count = 0

    def get_native_compile_guard_state(self, forward_batch, layer_id):
        del forward_batch, layer_id
        self.call_count += 1
        return self.supported, self.supported

    def supports_native_compile_for_batch(self, forward_batch, layer_id):
        return self.get_native_compile_guard_state(forward_batch, layer_id)[0]


class FakeGuardedNativeIndexer(MultiPlatformOp):
    def __init__(self, index_topk: int):
        super().__init__()
        self.index_topk = index_topk
        self.eager_call_count = 0
        self.native_call_count = 0

    def dispatch_forward(self):
        return self.forward_cpu

    def get_native_compile_guard_state(self, forward_batch, layer_id):
        del layer_id
        max_kv_len = forward_batch.attn_backend.forward_metadata.max_seq_len_k
        supported = max_kv_len <= self.index_topk
        return supported, True

    def supports_native_compile_for_batch(self, forward_batch, layer_id):
        return self.get_native_compile_guard_state(forward_batch, layer_id)[0]

    def _forward_eager(self, forward_batch):
        self.eager_call_count += 1
        return torch.zeros((forward_batch.extend_num_tokens, 1), dtype=torch.int32)

    def forward_native(self, forward_batch):
        self.native_call_count += 1
        max_kv_len = forward_batch.attn_backend.forward_metadata.max_seq_len_k
        if max_kv_len > self.index_topk:
            raise NotImplementedError(
                "Indexer forward_native only supports the short-ISL k-only path."
            )
        return self._forward_eager(forward_batch)

    def forward_cuda(self, forward_batch):
        return self._forward_eager(forward_batch)

    def forward_cpu(self, forward_batch):
        return self._forward_eager(forward_batch)


class FakeNSACompileSelfAttn(nn.Module):
    def __init__(
        self,
        *,
        indexer,
        skip_topk: bool = False,
        next_skip_topk: bool = False,
    ):
        super().__init__()
        self.use_nsa = True
        self.q_lora_rank = 1
        self.skip_topk = skip_topk
        self.next_skip_topk = next_skip_topk
        self.indexer = indexer


class FakeNSACompileLayer(FakePrefillCompileLayer):
    def __init__(
        self,
        hidden_size: int,
        layer_id: int,
        *,
        indexer_supported: bool,
        skip_topk: bool = False,
        next_skip_topk: bool = False,
    ):
        super().__init__(hidden_size, layer_id)
        self.self_attn = FakeNSACompileSelfAttn(
            indexer=FakeIndexer(indexer_supported),
            skip_topk=skip_topk,
            next_skip_topk=next_skip_topk,
        )


class FakeNSAFallbackRegressionLayer(FakePrefillCompileLayer):
    def __init__(
        self,
        hidden_size: int,
        layer_id: int,
        *,
        index_topk: int,
    ):
        super().__init__(hidden_size, layer_id)
        self.self_attn = FakeNSACompileSelfAttn(
            indexer=FakeGuardedNativeIndexer(index_topk)
        )

    def _forward_prefill_impl(
        self,
        positions,
        hidden_states,
        forward_batch,
        residual,
        zero_allocator,
        gemm_output_zero_allocator,
        llama_4_scaling,
        prev_topk_indices=None,
        should_allreduce_fusion=False,
        use_reduce_scatter=False,
    ):
        hidden_states, residual, _ = super()._forward_prefill_impl(
            positions,
            hidden_states,
            forward_batch,
            residual,
            zero_allocator,
            gemm_output_zero_allocator,
            llama_4_scaling,
            prev_topk_indices=prev_topk_indices,
            should_allreduce_fusion=should_allreduce_fusion,
            use_reduce_scatter=use_reduce_scatter,
        )
        topk_indices = self.self_attn.indexer(forward_batch)
        return hidden_states, residual, topk_indices


class FakeLogitsProcessor(nn.Module):
    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head,
        forward_batch,
        aux_hidden_states=None,
    ):
        del input_ids, aux_hidden_states
        last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        return LogitsProcessorOutput(
            next_token_logits=lm_head(hidden_states[last_index])
        )


class FakeAttnTpContext:
    def maybe_input_scattered(self, forward_batch):
        del forward_batch
        return nullcontext()


def _build_forward_batch(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    forward_mode: ForwardMode = ForwardMode.EXTEND,
) -> ForwardBatch:
    seq_len = input_ids.numel()
    return ForwardBatch(
        forward_mode=forward_mode,
        batch_size=1,
        input_ids=input_ids,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        out_cache_loc=torch.arange(seq_len, dtype=torch.int32),
        seq_lens_sum=seq_len,
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        positions=positions,
        extend_num_tokens=seq_len,
        extend_seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        extend_seq_lens_cpu=[seq_len],
        return_logprob=False,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def _build_tiny_compile_test_model(
    *, num_layers: int = 1, first_k_dense_replace: int = 0
) -> DeepseekV2ForCausalLM:
    hidden_size = 16
    vocab_size = 32

    model = object.__new__(DeepseekV2Model)
    nn.Module.__init__(model)
    model.padding_id = 0
    model.vocab_size = vocab_size
    model.first_k_dense_replace = first_k_dense_replace
    model.pp_group = FakePPGroup()
    model.nsa_enable_prefill_cp = False
    model.cp_size = None
    model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
    model.alt_stream = None
    model.layers = nn.ModuleList(
        [FakePrefillCompileLayer(hidden_size, layer_id=i) for i in range(num_layers)]
    )
    model.start_layer = 0
    model.end_layer = num_layers
    model.norm = nn.LayerNorm(hidden_size)
    model.gemm_output_zero_allocator_size = 0
    model.layers_to_capture = []
    model.enable_a2a_moe = False
    model.llama_4_scaling_config = None
    model._experimental_prefill_layer_compile_groups = None
    model._experimental_prefill_layer_compile_groups_by_start = {}
    model._experimental_prefill_layer_compile_group_size = None
    model._experimental_prefill_layer_compile_group_start_cached = None

    causal_lm = object.__new__(DeepseekV2ForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.nsa_enable_prefill_cp = False
    causal_lm.cp_rank = None
    causal_lm.cp_size = None
    causal_lm.use_nsa = False
    causal_lm.pp_group = FakePPGroup()
    causal_lm.model = model
    causal_lm.capture_aux_hidden_states = False
    causal_lm.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    causal_lm.logits_processor = FakeLogitsProcessor()
    return causal_lm


def _build_fake_nsa_attn_backend(*, metadata_present: bool, max_seq_len_k: int):
    forward_metadata = SimpleNamespace(max_seq_len_k=max_seq_len_k)

    def get_indexer_metadata(layer_id, forward_batch):
        del layer_id, forward_batch
        if not metadata_present:
            return None
        return SimpleNamespace(attn_metadata=forward_metadata)

    return SimpleNamespace(
        get_indexer_metadata=get_indexer_metadata,
        forward_metadata=forward_metadata,
    )


class TestDeepseekV2NativeCompileGuards(unittest.TestCase):
    def test_rejects_gather_output(self):
        module = FakeNativeCompileLinear(gather_output=True)

        with self.assertRaisesRegex(RuntimeError, "gather_output=True"):
            _validate_native_compile_linear_semantics(
                module, module_name="q_proj", allow_bias=False
            )

    def test_rejects_input_split_inside_row_parallel_wrapper(self):
        module = FakeNativeCompileLinear(input_is_parallel=False)

        with self.assertRaisesRegex(RuntimeError, "input_is_parallel=False"):
            _validate_native_compile_linear_semantics(
                module, module_name="o_proj", allow_bias=True
            )

    def test_rejects_skip_bias_add(self):
        module = FakeNativeCompileLinear(skip_bias_add=True)

        with self.assertRaisesRegex(RuntimeError, "skip_bias_add=True"):
            _validate_native_compile_linear_semantics(
                module, module_name="q_proj", allow_bias=False
            )

    def test_allows_bias_only_when_explicitly_supported(self):
        module = FakeNativeCompileLinear(bias=True)

        with self.assertRaisesRegex(RuntimeError, "with bias"):
            _validate_native_compile_linear_semantics(
                module, module_name="down_proj", allow_bias=False
            )

        _validate_native_compile_linear_semantics(
            module, module_name="o_proj", allow_bias=True
        )


class TestDeepseekV2NativeCompileFlags(unittest.TestCase):
    def test_mlp_bf16_gemm_flag_defaults_enabled(self):
        envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_DISABLE_MLP_BF16_GEMMS.clear()

        self.assertTrue(_deepseek_prefill_mlp_native_bf16_gemms_enabled())

    def test_mlp_bf16_gemm_flag_can_disable_native_path(self):
        mlp = object.__new__(DeepseekV2MLP)
        object.__setattr__(mlp, "gate_up_proj", object())
        object.__setattr__(mlp, "down_proj", object())

        with patch.object(
            DeepseekV2MLP,
            "_is_supported_block_fp8_linear_native_compile_module",
            return_value=True,
        ):
            with envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_DISABLE_MLP_BF16_GEMMS.override(
                True
            ):
                self.assertFalse(_deepseek_prefill_mlp_native_bf16_gemms_enabled())
                self.assertFalse(mlp._should_use_native_bf16_compile_path())

            self.assertTrue(_deepseek_prefill_mlp_native_bf16_gemms_enabled())
            self.assertTrue(mlp._should_use_native_bf16_compile_path())

    def test_prepare_compile_state_mirrors_kv_handles_to_attn_mqa(self):
        kv_buffer_storage = torch.empty((4, 8), dtype=torch.float32)
        token_to_kv_pool = SimpleNamespace(
            dtype=torch.float16,
            store_dtype=torch.uint8,
            nsa_kv_cache_store_fp8=True,
            use_nsa=True,
            get_key_buffer_storage=lambda layer_id: (
                self.assertEqual(layer_id, 7) or kv_buffer_storage
            ),
        )
        forward_batch = SimpleNamespace(token_to_kv_pool=token_to_kv_pool)

        layer = object.__new__(DeepseekV2DecoderLayer)
        object.__setattr__(layer, "layer_id", 7)
        object.__setattr__(
            layer,
            "self_attn",
            SimpleNamespace(attn_mqa=SimpleNamespace()),
        )
        object.__setattr__(layer, "mlp", SimpleNamespace())

        with patch(
            "sglang.srt.models.deepseek_v2._prewarm_flashinfer_lazy_modules_for_experimental_prefill_compile",
            lambda: None,
        ):
            DeepseekV2DecoderLayer._prepare_experimental_prefill_compile_state(
                layer, forward_batch
            )

        for target in (layer.self_attn, layer.self_attn.attn_mqa):
            self.assertIs(
                target._experimental_prefill_kv_buffer_storage, kv_buffer_storage
            )
            self.assertEqual(target._experimental_prefill_kv_cache_dtype, torch.float16)
            self.assertEqual(target._experimental_prefill_kv_store_dtype, torch.uint8)
            self.assertTrue(target._experimental_prefill_nsa_kv_cache_store_fp8)
            self.assertTrue(target._experimental_prefill_use_nsa)

    def test_cached_mla_kv_buffer_view_is_derived_from_storage(self):
        kv_buffer_storage = torch.empty((4, 8), dtype=torch.uint8)
        layer = SimpleNamespace(
            _experimental_prefill_kv_buffer_storage=kv_buffer_storage,
            _experimental_prefill_kv_cache_dtype=torch.float16,
            _experimental_prefill_kv_store_dtype=torch.uint8,
        )

        kv_buffer_view = _get_cached_mla_kv_buffer_view(layer)

        self.assertEqual(kv_buffer_view.dtype, torch.float16)
        self.assertEqual(kv_buffer_view.data_ptr(), kv_buffer_storage.data_ptr())

    def test_dynamic_input_marking_includes_nsa_page_tables(self):
        positions = torch.arange(4, dtype=torch.int32)
        hidden_states = torch.randn(4, 8)
        page_table_1 = torch.zeros((2, 16), dtype=torch.int32)
        real_page_table = torch.zeros((2, 4), dtype=torch.int32)
        page_table_1_flattened = torch.zeros(7, dtype=torch.int32)
        token_to_batch_idx = torch.zeros(7, dtype=torch.int32)
        nsa_seqlens_expanded = torch.zeros(7, dtype=torch.int32)
        forward_batch = SimpleNamespace(
            attn_backend=SimpleNamespace(
                forward_metadata=SimpleNamespace(
                    page_table_1=page_table_1,
                    real_page_table=real_page_table,
                    page_table_1_flattened=page_table_1_flattened,
                    token_to_batch_idx=token_to_batch_idx,
                    nsa_seqlens_expanded=nsa_seqlens_expanded,
                )
            )
        )

        seen = []

        def fake_maybe_mark_dynamic(tensor, dims):
            seen.append((tensor, tuple(dims)))

        with patch(
            "sglang.srt.models.deepseek_v2.torch._dynamo.maybe_mark_dynamic",
            side_effect=fake_maybe_mark_dynamic,
        ):
            _mark_experimental_prefill_dynamic_inputs(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        marked_tensors = {tensor for tensor, dims in seen if dims == (0,)}
        self.assertIn(positions, marked_tensors)
        self.assertIn(hidden_states, marked_tensors)
        self.assertIn(page_table_1, marked_tensors)
        self.assertIn(real_page_table, marked_tensors)
        self.assertIn(page_table_1_flattened, marked_tensors)
        self.assertIn(token_to_batch_idx, marked_tensors)
        self.assertIn(nsa_seqlens_expanded, marked_tensors)


class TestMultiPlatformCompileCallables(unittest.TestCase):
    def test_reuses_compile_callable_across_reentry(self):
        op = FakeCompileModeOp()
        eager_method = op._forward_method

        op.enter_torch_compile(num_tokens=4)
        first_compile_method = op._forward_method

        op.leave_torch_compile()
        op.enter_torch_compile(num_tokens=4)
        second_compile_method = op._forward_method

        self.assertIsNot(first_compile_method, eager_method)
        self.assertIs(first_compile_method, second_compile_method)

    def test_topk_keeps_cuda_forward_for_flashinfer_trtllm(self):
        op = FakeCompileModeTopK()
        eager_method = op._forward_method

        with patch(
            "sglang.srt.layers.moe.get_moe_runner_backend",
            return_value=SimpleNamespace(is_flashinfer_trtllm=lambda: True),
        ):
            op.enter_torch_compile(num_tokens=1)

        self.assertIs(op._forward_method, eager_method)

    def test_fused_moe_keeps_cuda_forward_for_flashinfer_trtllm(self):
        op = FakeCompileModeFusedMoE()
        eager_method = op._forward_method

        with patch(
            "sglang.srt.layers.moe.get_moe_runner_backend",
            return_value=SimpleNamespace(is_flashinfer_trtllm=lambda: True),
        ):
            op.enter_torch_compile(num_tokens=1)

        self.assertIs(op._forward_method, eager_method)


class TestDeepseekV2CompileModeTransitions(unittest.TestCase):
    def test_layer_groups_restore_compile_mode_after_decode_fallback(self):
        torch.manual_seed(0)
        model = _build_tiny_compile_test_model(
            num_layers=4, first_k_dense_replace=0
        ).model
        prefill_input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        prefill_positions = torch.arange(prefill_input_ids.numel(), dtype=torch.long)
        prefill_batch = _build_forward_batch(prefill_input_ids, prefill_positions)

        decode_input_ids = torch.tensor([5], dtype=torch.long)
        decode_positions = torch.tensor([4], dtype=torch.long)
        decode_batch = _build_forward_batch(
            decode_input_ids,
            decode_positions,
            forward_mode=ForwardMode.DECODE,
        )

        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_LAYER_GROUP_SIZE.override(
                    2
                )
            )
            stack.enter_context(
                patch("sglang.srt.models.deepseek_v2._is_cuda", True)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.nsa_use_prefill_cp",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_global_server_args",
                    return_value=SimpleNamespace(disable_piecewise_cuda_graph=False),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_moe_runner_backend",
                    return_value=SimpleNamespace(is_flashinfer_trtllm=lambda: True),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2._prewarm_flashinfer_lazy_modules_for_experimental_prefill_compile",
                    lambda: None,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2._get_experimental_prefill_compile_options",
                    lambda: {},
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_compiler_backend",
                    return_value="aot_eager",
                )
            )

            model.forward(prefill_input_ids, prefill_positions, prefill_batch)
            first_group = model._experimental_prefill_layer_compile_groups_by_start[0]
            second_group = model._experimental_prefill_layer_compile_groups_by_start[2]
            self.assertTrue(first_group._experimental_prefill_compile_enabled)
            self.assertTrue(second_group._experimental_prefill_compile_enabled)
            compile_methods = [
                layer.compile_mode_op._forward_method for layer in model.layers
            ]

            model.forward(decode_input_ids, decode_positions, decode_batch)

        self.assertTrue(first_group._experimental_prefill_compile_enabled)
        self.assertTrue(second_group._experimental_prefill_compile_enabled)
        for layer, compile_method in zip(model.layers, compile_methods):
            self.assertTrue(layer.compile_mode_op.is_torch_compile)
            self.assertIs(layer.compile_mode_op._forward_method, compile_method)


class TestDeepseekV2LayerGroupCompileGuards(unittest.TestCase):
    def test_fallback_temporarily_leaves_compile_mode_for_long_isl_nsa(self):
        torch.manual_seed(0)
        positions = torch.arange(4, dtype=torch.long)
        hidden_states = torch.randn(4, 16)
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        forward_batch = _build_forward_batch(input_ids, positions)
        forward_batch.attn_backend = _build_fake_nsa_attn_backend(
            metadata_present=True,
            max_seq_len_k=4096,
        )

        group = _ExperimentalPrefillCompileLayerGroup(
            layers=[
                FakeNSAFallbackRegressionLayer(
                    16,
                    layer_id=0,
                    index_topk=2048,
                ),
                FakePrefillCompileLayer(16, layer_id=1),
            ],
            start_layer=0,
            end_layer=2,
        )
        group._experimental_prefill_compiled_runner = object()
        group._enter_experimental_prefill_compile_mode(num_tokens=1)
        indexer = group.layers[0].self_attn.indexer

        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_LAYER_GROUP_SIZE.override(
                    2
                )
            )
            stack.enter_context(
                patch("sglang.srt.models.deepseek_v2._is_cuda", True)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.nsa_use_prefill_cp",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_moe_a2a_backend",
                    return_value=SimpleNamespace(is_none=lambda: True),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_moe_runner_backend",
                    return_value=SimpleNamespace(is_flashinfer_trtllm=lambda: True),
                )
            )
            actual = group.forward(
                positions,
                hidden_states,
                forward_batch,
                None,
                None,
            )

        self.assertEqual(indexer.eager_call_count, 1)
        self.assertEqual(indexer.native_call_count, 0)
        self.assertTrue(indexer.is_torch_compile)
        self.assertTrue(group._experimental_prefill_compile_enabled)
        self.assertEqual(actual[0].shape, hidden_states.shape)

    def test_skips_compiled_runner_for_unsupported_nsa_indexer_native_path(self):
        torch.manual_seed(0)
        positions = torch.arange(4, dtype=torch.long)
        hidden_states = torch.randn(4, 16)
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        forward_batch = _build_forward_batch(input_ids, positions)
        forward_batch.attn_backend = _build_fake_nsa_attn_backend(
            metadata_present=True,
            max_seq_len_k=8,
        )

        group = _ExperimentalPrefillCompileLayerGroup(
            layers=[
                FakeNSACompileLayer(16, layer_id=0, indexer_supported=False),
                FakePrefillCompileLayer(16, layer_id=1),
            ],
            start_layer=0,
            end_layer=2,
        )
        expected = group._forward_prefill_group_impl(
            positions,
            hidden_states,
            forward_batch,
            None,
            None,
        )

        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_LAYER_GROUP_SIZE.override(
                    2
                )
            )
            stack.enter_context(
                patch("sglang.srt.models.deepseek_v2._is_cuda", True)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.nsa_use_prefill_cp",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_moe_a2a_backend",
                    return_value=SimpleNamespace(is_none=lambda: True),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_moe_runner_backend",
                    return_value=SimpleNamespace(is_flashinfer_trtllm=lambda: True),
                )
            )
            actual = group.forward(
                positions,
                hidden_states,
                forward_batch,
                None,
                None,
            )

        self.assertIsNone(group._experimental_prefill_compiled_runner)
        torch.testing.assert_close(actual[0], expected[0], atol=1e-5, rtol=1e-5)
        self.assertIs(actual[1], expected[1])
        self.assertIs(actual[2], expected[2])

    def test_caches_unsupported_nsa_guard_result(self):
        torch.manual_seed(0)
        positions = torch.arange(4, dtype=torch.long)
        hidden_states = torch.randn(4, 16)
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        forward_batch = _build_forward_batch(input_ids, positions)
        forward_batch.attn_backend = _build_fake_nsa_attn_backend(
            metadata_present=True,
            max_seq_len_k=8,
        )

        group = _ExperimentalPrefillCompileLayerGroup(
            layers=[
                FakeNSACompileLayer(16, layer_id=0, indexer_supported=False),
                FakePrefillCompileLayer(16, layer_id=1),
            ],
            start_layer=0,
            end_layer=2,
        )
        fake_indexer = group.layers[0].self_attn.indexer

        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_LAYER_GROUP_SIZE.override(
                    2
                )
            )
            stack.enter_context(
                patch("sglang.srt.models.deepseek_v2._is_cuda", True)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.nsa_use_prefill_cp",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_moe_a2a_backend",
                    return_value=SimpleNamespace(is_none=lambda: True),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_moe_runner_backend",
                    return_value=SimpleNamespace(is_flashinfer_trtllm=lambda: True),
                )
            )
            group.forward(
                positions,
                hidden_states,
                forward_batch,
                None,
                None,
            )
            group.forward(
                positions,
                hidden_states,
                forward_batch,
                None,
                None,
            )

        self.assertEqual(fake_indexer.call_count, 1)

    def test_allows_compiled_runner_when_prev_topk_skips_unsupported_indexer(self):
        torch.manual_seed(0)
        positions = torch.arange(4, dtype=torch.long)
        hidden_states = torch.randn(4, 16)
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        forward_batch = _build_forward_batch(input_ids, positions)
        forward_batch.attn_backend = _build_fake_nsa_attn_backend(
            metadata_present=True,
            max_seq_len_k=8,
        )

        group = _ExperimentalPrefillCompileLayerGroup(
            layers=[
                FakeNSACompileLayer(
                    16,
                    layer_id=0,
                    indexer_supported=True,
                    next_skip_topk=True,
                ),
                FakeNSACompileLayer(
                    16,
                    layer_id=1,
                    indexer_supported=False,
                    skip_topk=True,
                ),
            ],
            start_layer=0,
            end_layer=2,
        )

        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_LAYER_GROUP_SIZE.override(
                    2
                )
            )
            stack.enter_context(
                patch("sglang.srt.models.deepseek_v2._is_cuda", True)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.nsa_use_prefill_cp",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_moe_a2a_backend",
                    return_value=SimpleNamespace(is_none=lambda: True),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_moe_runner_backend",
                    return_value=SimpleNamespace(is_flashinfer_trtllm=lambda: True),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2._prewarm_flashinfer_lazy_modules_for_experimental_prefill_compile",
                    lambda: None,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2._get_experimental_prefill_compile_options",
                    lambda: {},
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_compiler_backend",
                    return_value="aot_eager",
                )
            )
            group.forward(
                positions,
                hidden_states,
                forward_batch,
                None,
                None,
            )

        self.assertIsNotNone(group._experimental_prefill_compiled_runner)


class TestDeepseekV2LayerCompileGrouping(unittest.TestCase):
    def test_group_builder_starts_at_zero_by_default_and_keeps_tail_group(self):
        model = _build_tiny_compile_test_model(
            num_layers=8, first_k_dense_replace=3
        ).model

        with envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_LAYER_GROUP_SIZE.override(
            2
        ):
            groups = model._get_experimental_prefill_layer_compile_groups()

        self.assertEqual(
            [(group.start_layer, group.end_layer) for group in groups],
            [(0, 2), (2, 4), (4, 6), (6, 8)],
        )
        self.assertEqual(
            [group.layer_ids for group in groups],
            [[0, 1], [2, 3], [4, 5], [6, 7]],
        )

    def test_group_builder_honors_env_start_index(self):
        model = _build_tiny_compile_test_model(
            num_layers=8, first_k_dense_replace=3
        ).model

        with ExitStack() as stack:
            stack.enter_context(
                envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_LAYER_GROUP_SIZE.override(
                    2
                )
            )
            stack.enter_context(
                envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_LAYER_GROUP_START.override(
                    3
                )
            )
            groups = model._get_experimental_prefill_layer_compile_groups()

        self.assertEqual(
            [(group.start_layer, group.end_layer) for group in groups],
            [(3, 5), (5, 7), (7, 8)],
        )
        self.assertEqual(
            [group.layer_ids for group in groups],
            [[3, 4], [5, 6], [7]],
        )


class TestDeepseekV2NativeCompileEquivalence(unittest.TestCase):
    def test_grouped_layer_prefill_compile_logits_match_eager(self):
        torch.manual_seed(0)
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        positions = torch.arange(input_ids.numel(), dtype=torch.long)
        forward_batch = _build_forward_batch(input_ids, positions)

        eager_model = _build_tiny_compile_test_model(
            num_layers=8, first_k_dense_replace=3
        )
        compiled_model = copy.deepcopy(eager_model)

        with ExitStack() as stack:
            stack.enter_context(
                patch("sglang.srt.models.deepseek_v2._is_cuda", True)
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_attn_tp_context",
                    return_value=FakeAttnTpContext(),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.nsa_use_prefill_cp",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_global_server_args",
                    return_value=SimpleNamespace(disable_piecewise_cuda_graph=False),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_moe_runner_backend",
                    return_value=SimpleNamespace(is_flashinfer_trtllm=lambda: True),
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2._prewarm_flashinfer_lazy_modules_for_experimental_prefill_compile",
                    lambda: None,
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2._get_experimental_prefill_compile_options",
                    lambda: {},
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.models.deepseek_v2.get_compiler_backend",
                    return_value="aot_eager",
                )
            )

            eager_output = eager_model.forward(input_ids, positions, forward_batch)

            with ExitStack() as compiled_stack:
                compiled_stack.enter_context(
                    envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_LAYER_GROUP_SIZE.override(
                        3
                    )
                )
                compiled_output = compiled_model.forward(
                    input_ids, positions, forward_batch
                )

        self.assertIsNotNone(
            compiled_model.model._experimental_prefill_layer_compile_groups_by_start[0]._experimental_prefill_compiled_runner
        )
        self.assertIsNotNone(
            compiled_model.model._experimental_prefill_layer_compile_groups_by_start[3]._experimental_prefill_compiled_runner
        )
        self.assertIsNotNone(
            compiled_model.model._experimental_prefill_layer_compile_groups_by_start[6]._experimental_prefill_compiled_runner
        )
        torch.testing.assert_close(
            compiled_output.next_token_logits,
            eager_output.next_token_logits,
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
