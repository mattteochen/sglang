"""
Repro for compile-time blowup on deep models: torch.compile traces all layers
at once, causing symbolic expressions to grow and sympy simplification to hang.
nested_compile_region fixes this by tracing each layer as a separate subgraph.

Usage:
  # Basic run
  python test.py

  # With hierarchical_compile logging to verify nested_compile_region works
  TORCH_LOGS='+hierarchical_compile' python test.py

  # With a timeout to avoid waiting forever on the baseline
  python test.py --timeout 120

  # Adjust layer count (default 48; the crash was 78)
  python test.py --num-layers 78
"""

import argparse
import signal
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMoELayer(nn.Module):
    """Simplified MoE with top-k routing -- fully vectorized, no data-dependent branching."""

    def __init__(self, d_model, num_experts=4, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, 4 * d_model, bias=False),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model, bias=False),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        # x: [tokens, d_model]
        scores = self.gate(x)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # Compute all experts then gather -- no data-dependent branching
        # [tokens, num_experts, d_model]
        all_expert_out = torch.stack(
            [expert(x.view(-1, x.size(-1))) for expert in self.experts], dim=1
        )
        # Gather top-k expert outputs: [tokens, top_k, d_model]
        selected = torch.gather(
            all_expert_out, 1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )
        return (weights.unsqueeze(-1) * selected).sum(dim=1)


class DeepSeekLikeBlock(nn.Module):
    def __init__(self, d_model, nhead, num_experts=4, top_k=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = SimpleMoELayer(d_model, num_experts, top_k)

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Flatten for MoE, then reshape back -- this creates the symbolic
        # shape expressions that compound across layers
        flat = self.ln2(x).view(B * S, D)
        x = x + self.moe(flat).view(B, S, D)
        return x


class DeepSeekLikeBlockWithRegion(DeepSeekLikeBlock):
    @torch.compiler.nested_compile_region
    def forward(self, x):
        return super().forward(x)


class StackedModel(nn.Module):
    def __init__(self, block_cls, num_layers, d_model, nhead):
        super().__init__()
        self.layers = nn.ModuleList(
            [block_cls(d_model, nhead) for _ in range(num_layers)]
        )
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


class CompileTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise CompileTimeout()


def timed_compile_and_run(model, x, label, timeout=None):
    torch._dynamo.reset()
    compiled = torch.compile(model, dynamic=True)

    if timeout and hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        t0 = time.perf_counter()
        out = compiled(x)
        torch.cuda.synchronize()
        compile_time = time.perf_counter() - t0
    except CompileTimeout:
        compile_time = float("inf")
        print(f"[{label}]")
        print(f"  TIMED OUT after {timeout}s (symbolic expression blowup)")
        print()
        return compile_time, float("inf")
    finally:
        if timeout and hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    t0 = time.perf_counter()
    out2 = compiled(x)
    torch.cuda.synchronize()
    run_time = time.perf_counter() - t0

    print(f"[{label}]")
    print(f"  First call (compile + run): {compile_time:.2f}s")
    print(f"  Second call (run only):     {run_time:.4f}s")
    print()
    return compile_time, run_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=None,
                        help="Seconds before aborting baseline compile (avoids waiting forever)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the baseline (no nested_compile_region) run")
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(42)
    x = torch.randn(args.batch, args.seq_len, args.d_model, device=device)

    print(f"Model: {args.num_layers}-layer MoE transformer, d_model={args.d_model}, nhead={args.nhead}")
    print(f"Input: batch={args.batch}, seq_len={args.seq_len}, dynamic=True")
    print(f"Device: {device}")
    print(f"Tip: run with TORCH_LOGS='+hierarchical_compile' to see reuse decisions")
    print()

    if not args.skip_baseline:
        model_baseline = StackedModel(
            DeepSeekLikeBlock, args.num_layers, args.d_model, args.nhead
        ).to(device)
        t_baseline, r_baseline = timed_compile_and_run(
            model_baseline, x, f"Baseline ({args.num_layers} layers, no nested_compile_region)",
            timeout=args.timeout,
        )
    else:
        print("[Baseline skipped]")
        print()
        t_baseline = None

    model_nested = StackedModel(
        DeepSeekLikeBlockWithRegion, args.num_layers, args.d_model, args.nhead
    ).to(device)
    t_nested, r_nested = timed_compile_and_run(
        model_nested, x, f"With nested_compile_region ({args.num_layers} layers)",
    )

    print("=" * 60)
    if t_baseline is not None and t_baseline != float("inf"):
        speedup = t_baseline / t_nested if t_nested > 0 else float("inf")
        print(f"Compile-time speedup: {speedup:.2f}x")
        print(f"  Baseline:  {t_baseline:.2f}s")
        print(f"  Nested:    {t_nested:.2f}s")
        print(f"  Saved:     {t_baseline - t_nested:.2f}s")
    elif t_baseline == float("inf"):
        print(f"Baseline timed out; nested_compile_region compiled in {t_nested:.2f}s")
    else:
        print(f"nested_compile_region compiled in {t_nested:.2f}s")


if __name__ == "__main__":
    main()
