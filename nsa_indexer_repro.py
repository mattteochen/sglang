#!/usr/bin/env python3
"""Standalone NSA Indexer forward_cuda repro — GLM-5-FP8 tensor shapes.

Tensor sizes traced from zai-org/GLM-5-FP8 config.json + nsa_indexer.py.
Does NOT require loading the model.  Runs real GPU kernels (deep_gemm,
act_quant, fused_store, hadamard, rope, etc.) with random weights.

Execution paths (extend / prefill):
  - ISL <= 2048 (index_topk): _forward_cuda_k_only fast path
      wk(x) → k_norm → RoPE → hadamard → fused_store_k_cache → dummy topk
      SKIPS: wq_b GEMM, weights_proj GEMM, act_quant, MQA logits
  - ISL >  2048: full path with ragged fp8_mqa_logits + real topk

For ISL=1024, the fast path is ALWAYS taken. Per-token work:
  ❶ wk:          Linear (6144 → 128)     ~1.6M params
  ❷ k_norm:      LayerNorm(128)
  ❸ rotary_emb:  RoPE on (T, 64)
  ❹ hadamard:    Hadamard transform on (T, 128)
  ❺ fused_store: bf16 → fp8 quant + scatter to paged buffer
  ❻ dummy topk:  fill [0..len-1, -1, -1, ...] — no GEMM

Usage:
  python nsa_indexer_repro.py                               # ISL=1024 prefill
  python nsa_indexer_repro.py -E 4096                       # ISL=4096 (full path)
  python nsa_indexer_repro.py -B 4 -E 1024                  # batch=4
  python nsa_indexer_repro.py --profile                     # torch profiler
  python nsa_indexer_repro.py --mode decode -S 4096         # decode path
  python nsa_indexer_repro.py --mode both -E 1024 -S 1024   # both paths
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import torch

# ── Pre-import patches ────────────────────────────────────────────────
from sglang.srt.layers import dp_attention as _dp_attn

_dp_attn.get_attention_tp_size = lambda: 1

import sglang.srt.distributed.parallel_state as _ps


class _FakeGroup:
    world_size = 1
    rank = 0


_ps._TP = _FakeGroup()
_ps._ATTN_TP = _FakeGroup()

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

_sa = ServerArgs(model_path="dummy")
_sa.enable_dp_attention = False
_sa.nsa_prefill_backend = "trtllm"
_sa.nsa_decode_backend = "trtllm"
set_global_server_args_for_scheduler(_sa)

# ── Real sglang imports ──────────────────────────────────────────────
from sglang.srt.layers.attention.nsa.nsa_indexer import (
    BaseIndexerMetadata,
    Indexer,
)
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.linear import LinearBase
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

# =====================================================================
# GLM-5-FP8 config constants  (from config.json)
# =====================================================================
HIDDEN_SIZE = 6144
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
ROPE_HEAD_DIM = 64  # qk_rope_head_dim
INDEX_TOPK = 2048
Q_LORA_RANK = 2048
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
MAX_POSITION_EMBEDDINGS = 202752
ROPE_THETA = 1_000_000.0
PAGE_SIZE = 64
BLOCK_SIZE = 128
SCALE_FMT = "ue8m0"
IS_NEOX_STYLE = False  # indexer_rope_interleave = true

DEVICE = "cuda"
DTYPE = torch.bfloat16


# =====================================================================
# Mock metadata (follows the real NSAIndexerMetadata contract)
# =====================================================================
class MockIndexerMetadata(BaseIndexerMetadata):
    def __init__(
        self,
        batch_size: int,
        seq_lens: List[int],
        extend_lens: Optional[List[int]] = None,
        *,
        topk: int = INDEX_TOPK,
    ):
        self.batch_size = batch_size
        self.seq_lens = seq_lens
        self._extend_lens = extend_lens if extend_lens is not None else seq_lens
        self.device = DEVICE

        # Pre-compute everything topk_transform needs so that the hot path
        # is a single sgl_kernel call with zero tensor-construction overhead.
        self._precomputed_topk_args = self._build_topk_args(topk)

    def get_seqlens_int32(self) -> torch.Tensor:
        return torch.tensor(self.seq_lens, dtype=torch.int32, device=self.device)

    def get_page_table_64(self) -> torch.Tensor:
        max_sl = max(self.seq_lens)
        nb = (max_sl + 63) // 64
        pt = torch.zeros(
            (self.batch_size, nb), dtype=torch.int32, device=self.device
        )
        page_counter = 0
        for i in range(self.batch_size):
            n = (self.seq_lens[i] + 63) // 64
            pt[i, :n] = torch.arange(
                page_counter, page_counter + n, device=self.device
            )
            page_counter += n
        return pt

    def get_page_table_1(self) -> torch.Tensor:
        max_sl = max(self.seq_lens)
        pt = torch.zeros(
            (self.batch_size, max_sl), dtype=torch.int32, device=self.device
        )
        off = 0
        for i in range(self.batch_size):
            pt[i, : self.seq_lens[i]] = torch.arange(
                off, off + self.seq_lens[i], device=self.device
            )
            off += self.seq_lens[i]
        return pt

    def get_seqlens_expanded(self) -> torch.Tensor:
        result: List[int] = []
        for sl in self.seq_lens:
            result.extend(range(1, sl + 1))
        return torch.tensor(result, dtype=torch.int32, device=self.device)

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ks_all, ke_all = [], []
        k_off = 0
        for sl in self.seq_lens:
            ks_all.append(
                torch.full((sl,), k_off, dtype=torch.int32, device=self.device)
            )
            ke_all.append(
                torch.arange(
                    k_off + 1, k_off + sl + 1, dtype=torch.int32, device=self.device
                )
            )
            k_off += sl
        return torch.cat(ks_all), torch.cat(ke_all)

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        return torch.tensor(self.seq_lens, dtype=torch.int32, device="cpu")

    def get_indexer_seq_len(self) -> torch.Tensor:
        return torch.tensor(self.seq_lens, dtype=torch.int32, device=self.device)

    def get_nsa_extend_len_cpu(self) -> List[int]:
        return list(self._extend_lens)

    def get_token_to_batch_idx(self) -> torch.Tensor:
        result: List[int] = []
        for idx, sl in enumerate(self.seq_lens):
            result.extend([idx] * sl)
        return torch.tensor(result, dtype=torch.int32, device=self.device)

    def _build_topk_args(self, topk: int):
        """Build all tensors needed by fast_topk_transform_fused once."""
        lengths = self.get_seqlens_expanded()

        page_table_1 = self.get_page_table_1()
        max_kv = page_table_1.shape[1]
        if max_kv < topk:
            pad = torch.full(
                (page_table_1.shape[0], topk - max_kv),
                -1,
                dtype=torch.int32,
                device=page_table_1.device,
            )
            page_table_1 = torch.cat([page_table_1, pad], dim=-1)

        cu_seqlens_q = torch.zeros(
            self.batch_size + 1, dtype=torch.int32, device=self.device
        )
        for i in range(self.batch_size):
            cu_seqlens_q[i + 1] = cu_seqlens_q[i] + self._extend_lens[i]

        return dict(
            lengths=lengths,
            page_table_size_1=page_table_1,
            cu_seqlens_q=cu_seqlens_q,
            topk=topk,
        )

    def topk_transform(self, logits, topk, **kwargs):
        from sgl_kernel import fast_topk_transform_fused

        p = self._precomputed_topk_args
        return fast_topk_transform_fused(
            score=logits,
            lengths=p["lengths"],
            page_table_size_1=p["page_table_size_1"],
            cu_seqlens_q=p["cu_seqlens_q"],
            topk=p["topk"],
            row_starts=None,
        )


class MockAttnBackend:
    """Returns MockIndexerMetadata with correct extend_lens per mode."""

    def get_indexer_metadata(self, layer_id, forward_batch):
        seq_lens = forward_batch.seq_lens_cpu.tolist()
        if forward_batch.forward_mode.is_decode_or_idle():
            extend_lens = [1] * forward_batch.batch_size
        else:
            el = forward_batch.extend_seq_lens_cpu
            extend_lens = list(el) if el is not None else seq_lens
        return MockIndexerMetadata(forward_batch.batch_size, seq_lens, extend_lens)


# =====================================================================
# Factory helpers
# =====================================================================
def create_indexer(device=DEVICE, dtype=DTYPE) -> Indexer:
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.layers.linear import ReplicatedLinear as _RL

    fp8_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    indexer = Indexer(
        hidden_size=HIDDEN_SIZE,
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        index_topk=INDEX_TOPK,
        q_lora_rank=Q_LORA_RANK,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rope_theta=ROPE_THETA,
        scale_fmt=SCALE_FMT,
        block_size=BLOCK_SIZE,
        is_neox_style=IS_NEOX_STYLE,
        layer_id=0,
        quant_config=None,
        alt_stream=None,
    )

    # Replace wk with FP8 block-quantized version to match real model.
    # wk output_dim=128 is divisible by block_n=128 ✓
    # (weights_proj output_dim=32 is NOT, so we keep it bf16 — it's unused
    # in the _forward_cuda_k_only fast path anyway.)
    indexer.wk = _RL(
        HIDDEN_SIZE,
        INDEX_HEAD_DIM,
        bias=False,
        quant_config=fp8_config,
    )
    torch.set_default_dtype(prev_dtype)

    indexer = indexer.to(device=device)

    # Init FP8 weight + block scales for wk
    wk = indexer.wk
    wk.weight.data = torch.randn(
        wk.weight.shape, device=device, dtype=dtype
    ).to(torch.float8_e4m3fn)
    wk.weight_scale_inv.data.fill_(1.0)
    wk.weight_scale_inv.format_ue8m0 = False
    wk.quant_method.process_weights_after_loading(wk)

    # Keep other linears (wq_b, weights_proj) in bf16 — unused in fast path
    for name, module in indexer.named_modules():
        if isinstance(module, LinearBase) and not isinstance(module, LayerNorm):
            if module is not indexer.wk:
                module.to(dtype=dtype)

    return indexer


def create_kv_pool(max_tokens: int, device=DEVICE) -> NSATokenToKVPool:
    return NSATokenToKVPool(
        size=max_tokens,
        page_size=PAGE_SIZE,
        dtype=torch.float8_e4m3fn,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=1,
        device=device,
        index_head_dim=INDEX_HEAD_DIM,
        enable_memory_saver=False,
        kv_cache_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
    )


def make_decode_batch(
    batch_size: int,
    kv_len: int,
    kv_pool: NSATokenToKVPool,
    device=DEVICE,
) -> ForwardBatch:
    """ForwardBatch for decode.  kv_len = total seq length including this step."""
    fb = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        input_ids=torch.randint(0, 100, (batch_size,), device=device),
        req_pool_indices=torch.arange(batch_size, device=device),
        seq_lens=torch.full((batch_size,), kv_len, device=device, dtype=torch.int32),
        out_cache_loc=torch.arange(
            (kv_len - 1) * batch_size,
            kv_len * batch_size,
            device=device,
            dtype=torch.int64,
        ),
        seq_lens_sum=batch_size * kv_len,
        seq_lens_cpu=torch.full((batch_size,), kv_len, dtype=torch.int32),
        attn_backend=MockAttnBackend(),
    )
    fb.token_to_kv_pool = kv_pool
    return fb


def make_extend_batch(
    batch_size: int,
    extend_len: int,
    kv_pool: NSATokenToKVPool,
    device=DEVICE,
) -> ForwardBatch:
    """ForwardBatch for extend (prefill).  No prefix — seq_lens == extend_lens."""
    total_tokens = batch_size * extend_len
    fb = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=batch_size,
        input_ids=torch.randint(0, 100, (total_tokens,), device=device),
        req_pool_indices=torch.arange(batch_size, device=device),
        seq_lens=torch.full(
            (batch_size,), extend_len, device=device, dtype=torch.int32
        ),
        out_cache_loc=torch.arange(total_tokens, device=device, dtype=torch.int64),
        seq_lens_sum=total_tokens,
        seq_lens_cpu=torch.full((batch_size,), extend_len, dtype=torch.int32),
        extend_prefix_lens=torch.zeros(batch_size, device=device, dtype=torch.int32),
        extend_prefix_lens_cpu=[0] * batch_size,
        extend_seq_lens=torch.full(
            (batch_size,), extend_len, device=device, dtype=torch.int32
        ),
        extend_seq_lens_cpu=[extend_len] * batch_size,
        attn_backend=MockAttnBackend(),
    )
    fb.token_to_kv_pool = kv_pool
    return fb


# =====================================================================
# Benchmark harness
# =====================================================================
def bench(fn, warmup=10, iters=100, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]
    print(f"  [{label}]  avg={avg:.3f} ms   p50={p50:.3f} ms   p99={p99:.3f} ms   (n={iters})")
    return times


def print_shapes_table(label, shapes):
    print(f"\n  {label}:")
    for name, shape, dtype in shapes:
        print(f"    {name:<26s} {str(shape):<30s} {dtype}")


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="NSA Indexer forward_cuda repro (GLM-5-FP8 shapes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-B", "--batch-size", type=int, default=1)
    parser.add_argument(
        "-S",
        "--kv-len",
        type=int,
        default=1024,
        help="Total KV length for decode (default 1024)",
    )
    parser.add_argument(
        "-E",
        "--extend-len",
        type=int,
        default=1024,
        help="Extend (prefill) length — ISL (default 1024)",
    )
    parser.add_argument(
        "--mode",
        choices=["decode", "extend", "both"],
        default="extend",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Export a Chrome trace (nsa_indexer_trace.json)",
    )
    args = parser.parse_args()

    B = args.batch_size
    device = DEVICE
    layer_id = 0

    max_tokens = B * max(args.kv_len, args.extend_len) + PAGE_SIZE * 2
    kv_pool = create_kv_pool(max_tokens, device)
    indexer = create_indexer(device, DTYPE)

    print()
    print("=" * 65)
    print("  GLM-5-FP8  NSA Indexer  forward_cuda  Repro")
    print("=" * 65)
    print(
        f"  hidden_size={HIDDEN_SIZE}  index_n_heads={INDEX_N_HEADS}  "
        f"index_head_dim={INDEX_HEAD_DIM}  rope_head_dim={ROPE_HEAD_DIM}"
    )
    print(
        f"  q_lora_rank={Q_LORA_RANK}  index_topk={INDEX_TOPK}  "
        f"page_size={PAGE_SIZE}  is_neox_style={IS_NEOX_STYLE}"
    )

    # ── EXTEND ───────────────────────────────────────────────────────
    if args.mode in ("extend", "both"):
        E = args.extend_len
        T = B * E
        max_kv_len = E  # no prefix, so seq_lens == extend_lens
        skip = max_kv_len <= INDEX_TOPK

        print(f"\n{'─'*65}")
        print(f"  EXTEND (prefill)   batch_size={B}  ISL={E}  total_tokens={T}")
        print(f"  skip_logits_computation={skip}  (max_kv_len={max_kv_len} {'<=' if skip else '>'} index_topk={INDEX_TOPK})")
        if skip:
            print(f"  → _forward_cuda_k_only fast path (skips wq_b, weights_proj, MQA logits)")

        print_shapes_table(
            "Input tensors",
            [
                ("x", f"({T}, {HIDDEN_SIZE})", "bf16"),
                ("q_lora", f"({T}, {Q_LORA_RANK})", "bf16"),
                ("positions", f"({T},)", "int64"),
            ],
        )
        if skip:
            print_shapes_table(
                "Fast path (_forward_cuda_k_only)",
                [
                    ("key (wk + k_norm)", f"({T}, {INDEX_HEAD_DIM})", "bf16"),
                    ("key (after hadamard)", f"({T}, {INDEX_HEAD_DIM})", "bf16"),
                    ("topk_result (dummy)", f"({T}, {INDEX_TOPK})", "int32"),
                ],
            )
        else:
            print_shapes_table(
                "Full ragged path",
                [
                    ("query", f"({T}, {INDEX_N_HEADS}, {INDEX_HEAD_DIM})", "bf16"),
                    ("key", f"({T}, {INDEX_HEAD_DIM})", "bf16"),
                    ("q_fp8", f"({T}, {INDEX_N_HEADS}, {INDEX_HEAD_DIM})", "fp8"),
                    ("q_scale", f"({T}, {INDEX_N_HEADS}, 1)", "fp32"),
                    ("weights", f"({T}, {INDEX_N_HEADS}, 1)", "fp32"),
                    ("logits", f"({T}, sum_kv_len)", "fp32"),
                    ("topk_result", f"({T}, {INDEX_TOPK})", "int32"),
                ],
            )

        x_ext = torch.randn(T, HIDDEN_SIZE, dtype=DTYPE, device=device)
        q_lora_ext = torch.randn(T, Q_LORA_RANK, dtype=DTYPE, device=device)
        positions_ext = torch.arange(T, device=device, dtype=torch.int64)
        fb_ext = make_extend_batch(B, E, kv_pool, device)

        def run_extend():
            return indexer(
                x=x_ext,
                q_lora=q_lora_ext,
                positions=positions_ext,
                forward_batch=fb_ext,
                layer_id=layer_id,
            )

        print()
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                record_shapes=True,
            ) as prof:
                for _ in range(args.warmup):
                    run_extend()
                torch.cuda.synchronize()
                for _ in range(args.iters):
                    run_extend()
                torch.cuda.synchronize()

            trace_path = "nsa_indexer_extend_trace.json"
            prof.export_chrome_trace(trace_path)
            print(f"\n  Chrome trace → {trace_path}")
            print(
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
            )
        else:
            bench(run_extend, args.warmup, args.iters, label="extend")

    # ── DECODE ───────────────────────────────────────────────────────
    if args.mode in ("decode", "both"):
        S = args.kv_len
        num_pages = (S + PAGE_SIZE - 1) // PAGE_SIZE
        max_sl = num_pages * PAGE_SIZE

        print(f"\n{'─'*65}")
        print(f"  DECODE   batch_size={B}  kv_len={S}")

        print_shapes_table(
            "Input tensors",
            [
                ("x", f"({B}, {HIDDEN_SIZE})", "bf16"),
                ("q_lora", f"({B}, {Q_LORA_RANK})", "bf16"),
                ("positions", f"({B},)", "int64"),
            ],
        )
        print_shapes_table(
            "Internal tensors",
            [
                ("query (wq_b + reshape)", f"({B}, {INDEX_N_HEADS}, {INDEX_HEAD_DIM})", "bf16"),
                ("key (wk + k_norm)", f"({B}, {INDEX_HEAD_DIM})", "bf16"),
                ("q_rope / k_rope", f"({B},{INDEX_N_HEADS},{ROPE_HEAD_DIM}) / ({B},{ROPE_HEAD_DIM})", "bf16"),
                ("q_fp8 (act_quant)", f"({B}, {INDEX_N_HEADS}, {INDEX_HEAD_DIM})", "fp8_e4m3fn"),
                ("q_scale", f"({B}, {INDEX_N_HEADS}, 1)", "fp32"),
                ("weights (head gate)", f"({B}, {INDEX_N_HEADS}, 1)", "fp32"),
            ],
        )
        print_shapes_table(
            "Paged MQA logits (_get_topk_paged)",
            [
                ("q_fp8 (unsqueezed)", f"({B}, 1, {INDEX_N_HEADS}, {INDEX_HEAD_DIM})", "fp8"),
                ("kv_cache (raw buf)", f"(P, {PAGE_SIZE}, 1, 132)", "uint8"),
                ("block_tables", f"({B}, {num_pages})", "int32"),
                ("seqlens", f"({B},)  values=[{S}]", "int32"),
                ("logits", f"({B}, {max_sl})", "fp32"),
                ("topk_result", f"({B}, {INDEX_TOPK})", "int32"),
            ],
        )

        # Seed the KV cache with an extend pass so pages contain valid fp8 data
        if args.mode == "both":
            print("\n  (KV cache seeded by extend above)")
        else:
            seed_len = min(S, 64)
            print(f"\n  Seeding KV cache with {seed_len}-token extend …")
            x_seed = torch.randn(B * seed_len, HIDDEN_SIZE, dtype=DTYPE, device=device)
            ql_seed = torch.randn(B * seed_len, Q_LORA_RANK, dtype=DTYPE, device=device)
            pos_seed = torch.arange(B * seed_len, device=device, dtype=torch.int64)
            fb_seed = make_extend_batch(B, seed_len, kv_pool, device)
            indexer(
                x=x_seed,
                q_lora=ql_seed,
                positions=pos_seed,
                forward_batch=fb_seed,
                layer_id=layer_id,
            )
            torch.cuda.synchronize()

        x_dec = torch.randn(B, HIDDEN_SIZE, dtype=DTYPE, device=device)
        q_lora_dec = torch.randn(B, Q_LORA_RANK, dtype=DTYPE, device=device)
        positions_dec = torch.full((B,), S - 1, device=device, dtype=torch.int64)
        fb_dec = make_decode_batch(B, S, kv_pool, device)

        def run_decode():
            return indexer(
                x=x_dec,
                q_lora=q_lora_dec,
                positions=positions_dec,
                forward_batch=fb_dec,
                layer_id=layer_id,
            )

        print()
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
                record_shapes=True,
            ) as prof:
                for _ in range(args.warmup):
                    run_decode()
                torch.cuda.synchronize()
                for _ in range(args.iters):
                    run_decode()
                torch.cuda.synchronize()

            trace_path = "nsa_indexer_trace.json"
            prof.export_chrome_trace(trace_path)
            print(f"\n  Chrome trace → {trace_path}")
            print(
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
            )
        else:
            bench(run_decode, args.warmup, args.iters, label="decode")

    print()


if __name__ == "__main__":
    main()
