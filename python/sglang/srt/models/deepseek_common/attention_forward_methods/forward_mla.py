from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_common.utils import (
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    _is_cpu,
    _is_cublas_ge_129,
    _is_cuda,
    _is_gfx95_supported,
    _is_hip,
    _use_aiter,
    _use_aiter_gfx95,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

if _is_cuda:
    from sgl_kernel import bmm_fp8 as _raw_bmm_fp8

    from sglang.srt.utils.custom_op import register_custom_op

    # TODO(yuwei): remove this wrapper after sgl-kernel registers its own fake/meta impl
    # Wrap bmm_fp8 as a custom op so torch.compile does not trace into
    # torch.cuda.current_blas_handle() (which returns a non-Tensor).
    @register_custom_op(mutates_args=["out"])
    def _bmm_fp8_op(
        A: torch.Tensor,
        B: torch.Tensor,
        out: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
    ) -> None:
        _raw_bmm_fp8(A, B, A_scale, B_scale, out.dtype, out)

    def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
        if out is None:
            out = torch.empty(
                (A.shape[0], A.shape[1], B.shape[2]),
                device=A.device,
                dtype=dtype,
            )
        _bmm_fp8_op(A, B, out, A_scale, B_scale)
        return out


if _use_aiter:
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
    )
if _use_aiter_gfx95:
    from aiter.ops.triton.fused_fp8_quant import (
        fused_flatten_fp8_group_quant,
        fused_rms_fp8_group_quant,
    )

    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
        fused_rms_mxfp4_quant,
    )
    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla


# ------------------------------------------------------------------ #
#  Compiled wide-boundary forward_absorb_prepare (FP8 k-only path)   #
# ------------------------------------------------------------------ #

if _is_cuda:
    from sglang.jit_kernel.fused_store_index_cache import fused_store_index_k_cache_op
    from sglang.jit_kernel.hadamard import hadamard_transform_op
    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb

    @torch.compile(
        dynamic=True,
        options={
            "triton.enable_pdl": True,
            "combo_kernels": True,
            "cpp_wrapper": True,
            "trace.enabled": True,
        },
    )
    def _forward_absorb_prepare_native_inner(
        qkv_latent: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        # Pre-dequanted BF16 weights
        q_b_proj_weight: torch.Tensor,
        wk_weight: torch.Tensor,
        w_kc_bf16: torch.Tensor,
        # Norm weights
        q_a_ln_weight: torch.Tensor,
        q_a_ln_eps: float,
        kv_a_ln_weight: torch.Tensor,
        kv_a_ln_eps: float,
        k_norm_weight: torch.Tensor,
        k_norm_bias: torch.Tensor,
        k_norm_eps: float,
        # RoPE caches
        mla_cos_sin_cache: torch.Tensor,
        mla_is_neox_style: bool,
        indexer_cos_sin_cache: torch.Tensor,
        indexer_is_neox_style: bool,
        indexer_head_size: int,
        indexer_rotary_dim: int,
        # Indexer K-cache args
        index_k_buf: torch.Tensor,
        out_cache_loc: torch.Tensor,
        page_size: int,
        # Indexer topk args (pre-computed outside compile boundary)
        seq_lens_expanded: torch.Tensor,
        # Shape constants
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        num_local_heads: int,
        qk_head_dim: int,
        qk_nope_head_dim: int,
        indexer_rope_head_dim: int,
        indexer_head_dim: int,
        index_topk: int,
        llama_4_scaling,
    ):
        """torch.compile-friendly wide-boundary forward_absorb_prepare.

        Replaces all FP8 GEMMs with BF16 matmuls, custom-op norms with
        native torch norms, and custom-op RoPE with native torch RoPE.
        The Hadamard transform and K-cache store remain as custom ops
        (opaque leaves in the compiled graph).
        """
        # 1. Split qkv latent
        q, latent_cache = qkv_latent.split(
            [q_lora_rank, kv_lora_rank + qk_rope_head_dim], dim=-1
        )
        k_nope = latent_cache[..., :kv_lora_rank]

        # 2. RMSNorm (native) — q_a_layernorm
        q_f32 = q.float()
        q_var = q_f32.pow(2).mean(-1, keepdim=True)
        q = q_f32 * torch.rsqrt(q_var + q_a_ln_eps)
        q = (q * q_a_ln_weight).to(torch.bfloat16)

        # 3. RMSNorm (native) — kv_a_layernorm
        kn_f32 = k_nope.float()
        kn_var = kn_f32.pow(2).mean(-1, keepdim=True)
        k_nope = kn_f32 * torch.rsqrt(kn_var + kv_a_ln_eps)
        k_nope = (k_nope * kv_a_ln_weight).to(torch.bfloat16)

        # 4. q_b_proj (BF16 matmul)
        q_out = (q @ q_b_proj_weight.t()).view(-1, num_local_heads, qk_head_dim)

        # 5. Indexer k-only path (inlined)
        # 5a. wk projection (BF16 matmul)
        key = hidden_states @ wk_weight.t()

        # 5b. LayerNorm (native) on key
        key_orig_dtype = key.dtype
        key_f32 = key.to(k_norm_weight.dtype)
        key_f32 = torch.nn.functional.layer_norm(
            key_f32,
            (indexer_head_dim,),
            weight=k_norm_weight,
            bias=k_norm_bias,
            eps=k_norm_eps,
        )
        key = key_f32.to(key_orig_dtype)

        # 5c. RoPE on key (native)
        k_rope = key[..., :indexer_rope_head_dim]
        cos_sin = indexer_cos_sin_cache.index_select(0, positions.flatten())
        cos, sin = cos_sin.chunk(2, dim=-1)
        num_tokens = positions.shape[0]
        k_rope_3d = k_rope.view(num_tokens, -1, indexer_head_size)
        k_rope_rot = k_rope_3d[..., :indexer_rotary_dim]
        k_rope_pass = k_rope_3d[..., indexer_rotary_dim:]
        k_rope_rot = apply_rotary_emb(k_rope_rot, cos, sin, indexer_is_neox_style)
        k_rope_out = torch.cat((k_rope_rot, k_rope_pass), dim=-1).reshape(
            k_rope.shape
        )
        key = torch.cat(
            [k_rope_out, key[..., indexer_rope_head_dim:]], dim=-1
        )

        # 5d. Hadamard transform (custom op — opaque leaf)
        key = hadamard_transform_op(key, indexer_head_dim**-0.5)

        # 5e. Store K cache (custom op)
        fused_store_index_k_cache_op(key, index_k_buf, out_cache_loc, page_size)

        # 5f. Dummy topk (pure torch)
        num_tokens_topk = seq_lens_expanded.shape[0]
        col_idx = torch.arange(index_topk, dtype=torch.int32, device=key.device)
        valid = col_idx.unsqueeze(0) < seq_lens_expanded.unsqueeze(1)
        neg_one = torch.tensor(-1, dtype=torch.int32, device=key.device)
        topk_indices = torch.where(
            valid, col_idx.unsqueeze(0).expand(num_tokens_topk, -1), neg_one
        )

        # 6. q_nope @ w_kc (BF16 BMM)
        q_nope, q_pe = q_out.split(
            [qk_nope_head_dim, qk_rope_head_dim], dim=-1
        )
        k_pe = latent_cache[..., kv_lora_rank:].unsqueeze(1)
        k_nope = k_nope.unsqueeze(1)

        q_nope_out = torch.bmm(
            q_nope.transpose(0, 1), w_kc_bf16
        ).transpose(0, 1)

        # 7. RoPE on q_pe, k_pe (native — MLA rotary)
        mla_cos_sin = mla_cos_sin_cache.index_select(0, positions.flatten())
        mla_cos, mla_sin = mla_cos_sin.chunk(2, dim=-1)

        # q_pe RoPE
        q_pe_shape = q_pe.shape
        q_pe_3d = q_pe.view(num_tokens, -1, q_pe.shape[-1])
        q_pe_rot = apply_rotary_emb(q_pe_3d, mla_cos, mla_sin, mla_is_neox_style)
        q_pe = q_pe_rot.reshape(q_pe_shape)

        # k_pe RoPE
        k_pe_shape = k_pe.shape
        k_pe_3d = k_pe.view(num_tokens, -1, k_pe.shape[-1])
        k_pe_rot = apply_rotary_emb(k_pe_3d, mla_cos, mla_sin, mla_is_neox_style)
        k_pe = k_pe_rot.reshape(k_pe_shape)

        return q_pe, k_pe, q_nope_out, k_nope, topk_indices, llama_4_scaling


class DeepseekMLAForwardMixin:

    def init_mla_forward(self: DeepseekV2AttentionMLA):
        self.flashinfer_mla_disable_ragged = (
            get_global_server_args().flashinfer_mla_disable_ragged
        )

    def forward_absorb_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        # -------------------------------------------------------------- #
        # Wide-compile fast path: compile the entire prepare when all     #
        # conditions are met (FP8 model, extend, k-only, NSA enabled).   #
        # -------------------------------------------------------------- #
        if (
            _is_cuda
            and envs.SGLANG_TORCH_COMPILE_INDEXER.get()
            and self.q_lora_rank is not None
            and self.use_nsa
            and forward_batch.forward_mode.is_extend()
            and not nsa_use_prefill_cp(forward_batch)
            and hasattr(self, "fused_qkv_a_proj_with_mqa")
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.float8_e4m3fn
            and forward_batch.seq_lens_cpu is not None
            and forward_batch.seq_lens_cpu.max().item() <= self.indexer.index_topk
        ):
            # All hoisted outside compile boundary:
            qkv_latent = get_attn_tp_context().fetch_qkv_latent()
            self._ensure_all_weights_bf16()
            metadata = forward_batch.attn_backend.get_indexer_metadata(
                self.layer_id, forward_batch
            )
            if metadata is not None:
                index_k_buf = (
                    forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
                        layer_id=self.layer_id
                    )
                )
                out_cache_loc = forward_batch.out_cache_loc
                if not out_cache_loc.is_contiguous():
                    out_cache_loc = out_cache_loc.contiguous()
                page_size = forward_batch.token_to_kv_pool.page_size

                q_pe, k_pe, q_nope_out, k_nope, topk_indices, llama_4_scaling = (
                    self._forward_absorb_prepare_native(
                        qkv_latent,
                        hidden_states,
                        positions,
                        forward_batch,
                        zero_allocator,
                        index_k_buf,
                        out_cache_loc,
                        page_size,
                        metadata,
                        llama_4_scaling,
                    )
                )

                return (
                    q_pe,
                    k_pe,
                    q_nope_out,
                    k_nope,
                    forward_batch,
                    zero_allocator,
                    positions,
                    topk_indices,
                    llama_4_scaling,
                )
            # metadata is None → fall through to the default path

        q_lora = None
        topk_indices = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                if _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                    q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                    )
                else:
                    q_lora = None
                    if (
                        _use_aiter_gfx95
                        and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):
                        if self.use_nsa:
                            q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=True,
                            )
                            q = q_quanted
                        else:
                            q, _, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=False,
                            )

                    else:
                        q = self.q_a_layernorm(q)
                        k_nope = self.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            if self.use_nsa:
                if q_lora is None:
                    q_lora = q

            # overlap q_b_proj and indexer during decode
            if (
                self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj(q)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                if not self.skip_topk or prev_topk_indices is None:
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                else:
                    topk_indices = prev_topk_indices
                current_stream.wait_stream(self.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                if q_lora is not None:
                    if not self.skip_topk or prev_topk_indices is None:
                        topk_indices = self.indexer(
                            x=hidden_states,
                            q_lora=q_lora,
                            positions=positions,
                            forward_batch=forward_batch,
                            layer_id=self.layer_id,
                        )
                    else:
                        topk_indices = prev_topk_indices
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_kc.transpose(-2, -1),
                    self.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                if (_use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn) or (
                    get_is_capture_mode() and self.w_kc.dtype == torch.float8_e4m3fnuz
                ):
                    # fp8 Triton kernel: always on gfx950,
                    # cudagraph-only on gfx942 (hides launch overhead)
                    q_nope_out = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=q_nope,
                        WQ=self.w_kc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,  # allocate (B, M, N)
                        transpose_bm=False,  # (B, M, N)
                        transpose_bm_in=True,  # (M, B, K)
                        dtype=torch.bfloat16,
                    )

                else:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )

        elif self.w_kc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                q_nope_out = torch.bmm(
                    q_nope.to(torch.bfloat16).transpose(0, 1),
                    self.w_kc.to(torch.bfloat16) * self.w_scale,
                )
            else:
                # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
                q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                    q_nope.transpose(0, 1),
                    (
                        torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                q_nope_out = bmm_fp8(
                    q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
                )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        skip_rope_for_nsa_tilelang_fused = self._skip_rope_for_nsa_tilelang_fused()
        if (
            self.rotary_emb is not None
            and (not self._fuse_rope_for_trtllm_mla(forward_batch))
            and (not skip_rope_for_nsa_tilelang_fused)
            and (not _use_aiter or not _is_gfx95_supported or self.use_nsa)
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if nsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
        )

    # ------------------------------------------------------------------ #
    #  torch.compile-friendly ("native") wide-compile path                #
    # ------------------------------------------------------------------ #

    def _ensure_all_weights_bf16(self: DeepseekV2AttentionMLA) -> None:
        """Lazily dequantize FP8 weights to bf16 for the wide-compile path.

        Called once on first use.  Stores dequanted copies as registered
        buffers so they persist across forward calls.

        Weights dequantized:
          - fused_qkv_a_proj_with_mqa  → _fused_qkv_weight_bf16
          - q_b_proj                   → _q_b_proj_weight_bf16
          - indexer.wk                 → indexer._wk_weight_bf16 (via existing helper)
          - w_kc                       → _w_kc_bf16 (if currently FP8)
        """
        from sglang.srt.layers.attention.nsa.nsa_indexer import (
            dequant_fp8_weight_to_bf16,
        )

        if hasattr(self, "_fused_qkv_weight_bf16"):
            return

        # --- fused_qkv_a_proj_with_mqa ---
        w = self.fused_qkv_a_proj_with_mqa.weight
        if w.dtype == torch.float8_e4m3fn:
            w_bf16 = dequant_fp8_weight_to_bf16(
                w, self.fused_qkv_a_proj_with_mqa.weight_scale_inv
            )
        else:
            w_bf16 = w.data.to(torch.bfloat16)
        self.register_buffer("_fused_qkv_weight_bf16", w_bf16)

        # --- q_b_proj ---
        w = self.q_b_proj.weight
        if w.dtype == torch.float8_e4m3fn:
            w_bf16 = dequant_fp8_weight_to_bf16(
                w, self.q_b_proj.weight_scale_inv
            )
        else:
            w_bf16 = w.data.to(torch.bfloat16)
        self.register_buffer("_q_b_proj_weight_bf16", w_bf16)

        # --- indexer.wk (delegate to existing helper) ---
        self.indexer._ensure_wk_weight_bf16()

        # --- w_kc (batched matmul weight) ---
        if self.w_kc is not None and self.w_kc.dtype == torch.float8_e4m3fn:
            # Tensor-quant FP8: w_kc (heads, qk_nope, kv_lora_rank) + w_scale
            w_kc_bf16 = self.w_kc.to(torch.bfloat16) * self.w_scale
            self.register_buffer("_w_kc_bf16", w_kc_bf16)
        elif self.w_kc is not None:
            # Already bf16 (from block_quant_dequant in post_load_weights)
            self.register_buffer("_w_kc_bf16", self.w_kc.to(torch.bfloat16))

    def _forward_absorb_prepare_native(
        self: DeepseekV2AttentionMLA,
        qkv_latent: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        index_k_buf: torch.Tensor,
        out_cache_loc: torch.Tensor,
        page_size: int,
        metadata,
        llama_4_scaling,
    ):
        """Dispatch wrapper that calls the compiled inner function.

        Hoists all buffer lookups and Python-level decisions outside
        the torch.compile boundary, then delegates to the compiled
        _forward_absorb_prepare_native_inner.
        """
        # Hoist metadata access outside compile boundary
        seq_lens_expanded = metadata.get_seqlens_expanded()

        q_pe, k_pe, q_nope_out, k_nope, topk_indices, llama_4_scaling = (
            _forward_absorb_prepare_native_inner(
                qkv_latent,
                hidden_states,
                positions,
                # Pre-dequanted BF16 weights
                self._q_b_proj_weight_bf16,
                self.indexer._wk_weight_bf16,
                self._w_kc_bf16,
                # Norm weights
                self.q_a_layernorm.weight,
                self.q_a_layernorm.variance_epsilon,
                self.kv_a_layernorm.weight,
                self.kv_a_layernorm.variance_epsilon,
                self.indexer.k_norm.weight,
                self.indexer.k_norm.bias,
                self.indexer.k_norm.variance_epsilon,
                # RoPE caches
                self.rotary_emb.cos_sin_cache,
                self.rotary_emb.is_neox_style,
                self.indexer.rotary_emb.cos_sin_cache,
                self.indexer.rotary_emb.is_neox_style,
                self.indexer.rotary_emb.head_size,
                self.indexer.rotary_emb.rotary_dim,
                # Indexer K-cache args
                index_k_buf,
                out_cache_loc,
                page_size,
                # Indexer topk args (pre-computed outside compile boundary)
                seq_lens_expanded,
                # Shape constants
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.num_local_heads,
                self.qk_head_dim,
                self.qk_nope_head_dim,
                self.indexer.rope_head_dim,
                self.indexer.head_dim,
                self.indexer.index_topk,
                llama_4_scaling,
            )
        )

        # Apply topk transform for PAGED/RAGGED modes (outside compile boundary)
        topk_indices = self.indexer._apply_topk_transform(
            topk_indices, seq_lens_expanded, metadata
        )

        return q_pe, k_pe, q_nope_out, k_nope, topk_indices, llama_4_scaling

    def forward_absorb_core(
        self: DeepseekV2AttentionMLA,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
    ):
        save_kv_cache = True

        if self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            if self._skip_rope_for_nsa_tilelang_fused() and self.rotary_emb is not None:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache
                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )
                q_cat, _, k_pe_fused, _ = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    forward_batch.token_to_kv_pool.get_key_buffer(
                        self.attn_mqa.layer_id
                    ),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )
                q_nope_fused = q_cat[..., : self.kv_lora_rank]
                q_pe_fused = q_cat[..., self.kv_lora_rank :]
                save_kv_cache = False
                if llama_4_scaling is not None:
                    q_nope_fused *= llama_4_scaling
                attn_output = self.attn_mqa(
                    q_nope_fused,
                    None,
                    None,
                    forward_batch,
                    q_rope=q_pe_fused,
                    k_rope=k_pe_fused,
                    save_kv_cache=save_kv_cache,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
            else:
                extra_args = {}
                if self._fuse_rope_for_trtllm_mla(forward_batch):
                    extra_args = {
                        "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                        "is_neox": self.rotary_emb.is_neox_style,
                        "llama_4_scaling": llama_4_scaling,
                    }
                attn_output = self.attn_mqa(
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
                    **extra_args,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
        else:
            if _use_aiter_gfx95:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache

                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )

                q, _, _, k = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    forward_batch.token_to_kv_pool.get_key_buffer(
                        self.attn_mqa.layer_id
                    ),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )

                save_kv_cache = False
            else:
                q = torch.cat([q_nope_out, q_pe], dim=-1)
                k = torch.cat([k_nope, k_pe], dim=-1)

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                attn_bmm_output = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_vc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:
                    attn_bmm_output = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=attn_output,
                        WQ=self.w_vc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,
                        transpose_bm=False,
                        transpose_bm_in=True,
                        dtype=torch.bfloat16,
                    )
                else:
                    attn_bmm_output = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.w_vc.to(torch.bfloat16) * self.w_scale,
                    )

            if self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_fp8_group_quant(
                    attn_bmm_output, group_size=128, dtype_quant=torch.float8_e4m3fn
                )
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        elif self.w_vc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
            else:
                attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                    attn_output.transpose(0, 1),
                    (
                        torch.zeros(
                            (1,), dtype=torch.float32, device=attn_output.device
                        )
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                attn_bmm_output = bmm_fp8(
                    attn_output_val,
                    self.w_vc,
                    attn_output_scale,
                    self.w_scale,
                    torch.bfloat16,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            if is_in_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
        output, _ = self.o_proj(attn_bmm_output)

        if self.next_skip_topk is None:
            return output

        # Return topk_indices for the next layer when enabling index cache
        if not self.next_skip_topk:
            return output, None
        else:
            return output, topk_indices

    def _fuse_rope_for_trtllm_mla(
        self: DeepseekV2AttentionMLA, forward_batch: ForwardBatch
    ) -> bool:
        """
        Check if we should skip rope and do fused rope+quantize for TRTLLM MLA decode in fp8_e4m3 path.
        """
        if self.current_attention_backend == "nsa":
            return (
                get_global_server_args().nsa_decode_backend == "trtllm"
                or get_global_server_args().nsa_prefill_backend == "trtllm"
            ) and forward_batch.attn_backend.kv_cache_dtype == torch.float8_e4m3fn

        return (
            self.current_attention_backend == "trtllm_mla"
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
            )
            and forward_batch.attn_backend.data_type == torch.float8_e4m3fn
        )

    def _skip_rope_for_nsa_tilelang_fused(self: DeepseekV2AttentionMLA) -> bool:
        """
        Check if we should skip rope and use fused rope+cache path for TileLang NSA on gfx95.
        """
        server_args = get_global_server_args()
        return (
            _use_aiter_gfx95
            and self.current_attention_backend == "nsa"
            and (
                server_args.nsa_decode_backend == "tilelang"
                or server_args.nsa_prefill_backend == "tilelang"
            )
        )
