import copy
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2Model,
    _disable_experimental_prefill_nested_compile_region,
    _deepseek_prefill_mlp_native_bf16_gemms_enabled,
    _enable_experimental_prefill_nested_compile_region,
    _get_experimental_prefill_nested_compile_region_target,
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


class FakePrefillCompileLayer(nn.Module):
    def __init__(self, hidden_size: int, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.layer_communicator = FakeLayerCommunicator()
        self.lin1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_scale = nn.Parameter(torch.randn(hidden_size))

    def _core(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        pos = positions.to(hidden_states.dtype).unsqueeze(-1) * self.pos_scale
        return self.lin2(torch.relu(self.lin1(hidden_states) + pos))

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
    input_ids: torch.Tensor, positions: torch.Tensor
) -> ForwardBatch:
    seq_len = input_ids.numel()
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
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
    *, num_layers: int = 1
) -> DeepseekV2ForCausalLM:
    hidden_size = 16
    vocab_size = 32

    model = object.__new__(DeepseekV2Model)
    nn.Module.__init__(model)
    model.padding_id = 0
    model.vocab_size = vocab_size
    model.first_k_dense_replace = 0
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
    model._experimental_prefill_compiled_runner = None
    model._experimental_prefill_compile_failed = False
    model._experimental_prefill_compile_enabled = False
    model._experimental_prefill_compile_logged_eligible = False
    model._experimental_prefill_compile_logged_success = False
    model._experimental_prefill_compile_warmup_count = 0

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


class TestDeepseekV2NestedCompileRegion(unittest.TestCase):
    def test_applies_only_from_layer_3(self):
        layers = [FakePrefillCompileLayer(hidden_size=8, layer_id=i) for i in range(6)]

        def fake_nested_compile_region(fn):
            def wrapped(*args, **kwargs):
                return fn(*args, **kwargs)

            return wrapped

        with patch(
            "sglang.srt.models.deepseek_v2._get_experimental_prefill_nested_compile_region_fn",
            side_effect=fake_nested_compile_region,
        ):
            with envs.SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_NESTED_COMPILE_REGION.override(
                True
            ):
                _enable_experimental_prefill_nested_compile_region()
                try:
                    targets = [
                        _get_experimental_prefill_nested_compile_region_target(layer)
                        for layer in layers
                    ]
                finally:
                    _disable_experimental_prefill_nested_compile_region()

        for layer_id in range(3):
            self.assertIs(
                targets[layer_id].__func__,
                layers[layer_id]._forward_prefill_impl.__func__,
            )
        for layer_id in range(3, len(layers)):
            self.assertIsNot(
                targets[layer_id].__func__,
                layers[layer_id]._forward_prefill_impl.__func__,
            )


class TestDeepseekV2NativeCompileEquivalence(unittest.TestCase):
    def test_prefill_compile_logits_match_eager_across_8_layers(self):
        torch.manual_seed(0)
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        positions = torch.arange(input_ids.numel(), dtype=torch.long)
        forward_batch = _build_forward_batch(input_ids, positions)

        eager_model = _build_tiny_compile_test_model(num_layers=8)
        compiled_model = copy.deepcopy(eager_model)

        with ExitStack() as stack:
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
            stack.enter_context(
                patch.object(
                    eager_model.model,
                    "_should_use_experimental_prefill_compile",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch.object(
                    compiled_model.model,
                    "_should_use_experimental_prefill_compile",
                    return_value=True,
                )
            )

            eager_output = eager_model.forward(input_ids, positions, forward_batch)
            compiled_output = compiled_model.forward(
                input_ids, positions, forward_batch
            )

        self.assertIsNotNone(compiled_model.model._experimental_prefill_compiled_runner)
        torch.testing.assert_close(
            compiled_output.next_token_logits,
            eager_output.next_token_logits,
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
