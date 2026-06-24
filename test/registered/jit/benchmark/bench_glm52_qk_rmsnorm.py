"""Benchmark GLM-5.2 MLA Q/K RMSNorm fusion.

Compares the torch.compile-generated fused path used by
``_native_qk_rmsnorm`` against running the two RMSNorms separately through
``RMSNorm.forward_cuda``.

Default dimensions match the GLM-5.2/DSA MLA normalization inputs observed in
the prefill path:

  - q: [num_tokens, 2048]
  - k: [num_tokens, 512]

Run:

    python3 test/registered/jit/benchmark/bench_glm52_qk_rmsnorm.py
    python3 test/registered/jit/benchmark/bench_glm52_qk_rmsnorm.py --tokens 32768
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import torch
import triton.testing

from sglang.srt.layers.layernorm import RMSNorm


DEFAULT_TOKENS = (1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_Q_DIM = 2048
DEFAULT_K_DIM = 512
DEFAULT_EPS = 1e-5
DEFAULT_RECOMPILE_LIMIT = 1024


# The production function in forward_mla.py is intentionally this small:
#
#   @torch.compile(dynamic=False, options={"triton.enable_pdl": True,
#                                          "combo_kernels": True,
#                                          "benchmark_combo_kernel": True})
#   def _native_qk_rmsnorm(q, k, q_norm, k_norm):
#       return q_norm.forward_native(q), k_norm.forward_native(k)
#
# Keep this benchmark standalone so it does not import the full MLA module or
# model stack while still measuring the same torch.compile fusion surface.
@torch.compile(
    dynamic=False,
    options={
        "triton.enable_pdl": True,
        "combo_kernels": True,
        "benchmark_combo_kernel": True,
    },
)
def _compiled_native_qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
):
    return q_norm.forward_native(q), k_norm.forward_native(k)


@dataclass(frozen=True)
class Timing:
    median_us: float
    p20_us: float
    p80_us: float


def _parse_dtype(name: str) -> torch.dtype:
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _make_norm(hidden_size: int, eps: float, dtype: torch.dtype, device: str) -> RMSNorm:
    norm = RMSNorm(hidden_size, eps=eps, weight_dtype=dtype).to(device=device)
    norm.weight.data.normal_(mean=1.0, std=0.01)
    return norm


def _bench(fn: Callable[[], object], *, use_cuda_graph: bool) -> Timing:
    # Exclude torch.compile and allocator warmup from the reported timing.
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    quantiles = [0.5, 0.2, 0.8]
    if use_cuda_graph:
        ms, p20_ms, p80_ms = triton.testing.do_bench_cudagraph(
            fn, quantiles=quantiles
        )
    else:
        ms, p20_ms, p80_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return Timing(ms * 1000, p20_ms * 1000, p80_ms * 1000)


def _check_close(compiled_out, cuda_out) -> float:
    q_compiled, k_compiled = compiled_out
    q_cuda, k_cuda = cuda_out
    q_diff = (q_compiled - q_cuda).abs().max().item()
    k_diff = (k_compiled - k_cuda).abs().max().item()
    torch.testing.assert_close(q_compiled, q_cuda, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_compiled, k_cuda, atol=1e-2, rtol=1e-2)
    return max(q_diff, k_diff)


def run_one(
    *,
    num_tokens: int,
    q_dim: int,
    k_dim: int,
    dtype: torch.dtype,
    device: str,
    eps: float,
    use_cuda_graph: bool,
    check: bool,
) -> tuple[Timing, Timing, float | None]:
    torch.manual_seed(0)
    q = torch.randn((num_tokens, q_dim), device=device, dtype=dtype)
    k = torch.randn((num_tokens, k_dim), device=device, dtype=dtype)
    q_norm = _make_norm(q_dim, eps, dtype, device)
    k_norm = _make_norm(k_dim, eps, dtype, device)

    def compiled_native():
        return _compiled_native_qk_rmsnorm(q, k, q_norm, k_norm)

    def separate_forward_cuda():
        return q_norm.forward_cuda(q), k_norm.forward_cuda(k)

    max_diff = None
    with torch.inference_mode():
        if check:
            max_diff = _check_close(compiled_native(), separate_forward_cuda())
            torch.cuda.synchronize()

        compiled_timing = _bench(compiled_native, use_cuda_graph=use_cuda_graph)
        cuda_timing = _bench(separate_forward_cuda, use_cuda_graph=use_cuda_graph)

    return compiled_timing, cuda_timing, max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--q-dim", type=int, default=DEFAULT_Q_DIM)
    parser.add_argument("--k-dim", type=int, default=DEFAULT_K_DIM)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--recompile-limit",
        type=int,
        default=DEFAULT_RECOMPILE_LIMIT,
        help=(
            "torch._dynamo.config.recompile_limit. dynamic=False specializes "
            "one graph per shape, so large token sweeps need this above 8."
        ),
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Use triton.testing.do_bench instead of do_bench_cudagraph.",
    )
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check compiled output against separate forward_cuda output.",
    )
    args = parser.parse_args()

    dtype = _parse_dtype(args.dtype)
    use_cuda_graph = not args.no_cuda_graph
    torch._dynamo.config.recompile_limit = args.recompile_limit

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this benchmark.")

    print(
        "GLM-5.2 Q/K RMSNorm benchmark: "
        f"q_dim={args.q_dim}, k_dim={args.k_dim}, dtype={dtype}, "
        f"eps={args.eps}, cuda_graph={use_cuda_graph}, "
        f"recompile_limit={args.recompile_limit}"
    )
    header = (
        f"{'tokens':>8}  {'compiled(us)':>13}  {'forward_cuda(us)':>16}  "
        f"{'speedup':>8}  {'max diff':>10}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for num_tokens in args.tokens:
        compiled_t, cuda_t, max_diff = run_one(
            num_tokens=num_tokens,
            q_dim=args.q_dim,
            k_dim=args.k_dim,
            dtype=dtype,
            device=args.device,
            eps=args.eps,
            use_cuda_graph=use_cuda_graph,
            check=args.check,
        )
        speedup = cuda_t.median_us / compiled_t.median_us
        diff_str = "n/a" if max_diff is None else f"{max_diff:.3e}"
        print(
            f"{num_tokens:8d}  "
            f"{compiled_t.median_us:13.3f}  "
            f"{cuda_t.median_us:16.3f}  "
            f"{speedup:8.2f}  "
            f"{diff_str:>10}"
        )
    print("-" * len(header))


if __name__ == "__main__":
    main()
