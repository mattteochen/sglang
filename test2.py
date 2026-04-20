"""
Repro for the case where nested_compile_region is present but reuse should not
help because every layer is structurally different.

Expected behavior with:
  TORCH_LOGS='+hierarchical_compile,recompiles' python test2.py --num-layers 10

- Dynamo should still emit hierarchical_compile logs if invoke_subgraph is used.
- We should see repeated subgraph installs and no cache-hit reuse pattern.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableMoELayer(nn.Module):
    """MoE where each layer can vary in expert count and hidden expansion."""

    def __init__(self, d_model, num_experts, top_k, expansion):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        hidden_dim = expansion * d_model
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim, bias=False),
                    nn.GELU(),
                    nn.Linear(hidden_dim, d_model, bias=False),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        scores = self.gate(x)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        all_expert_out = torch.stack(
            [expert(x.view(-1, x.size(-1))) for expert in self.experts], dim=1
        )
        selected = torch.gather(
            all_expert_out, 1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )
        return (weights.unsqueeze(-1) * selected).sum(dim=1)


def layer_config(layer_idx):
    # Make each block structurally distinct so invoke_subgraph reuse should fail.
    num_experts = 3 + (layer_idx % 4)
    expansion = 2 + (layer_idx % 5)
    top_k = min(2, num_experts)
    return num_experts, top_k, expansion


class HeterogeneousBlock(nn.Module):
    def __init__(self, layer_idx, d_model, nhead):
        super().__init__()
        num_experts, top_k, expansion = layer_config(layer_idx)
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = VariableMoELayer(d_model, num_experts, top_k, expansion)

    def forward(self, x):
        batch, seq_len, dim = x.shape
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        flat = self.ln2(x).view(batch * seq_len, dim)
        x = x + self.moe(flat).view(batch, seq_len, dim)
        return x


class HeterogeneousBlockWithRegion(HeterogeneousBlock):
    @torch.compiler.nested_compile_region(max_reuse_entries=32)
    def forward(self, x):
        return super().forward(x)


class HeterogeneousStackedModel(nn.Module):
    def __init__(self, block_cls, num_layers, d_model, nhead):
        super().__init__()
        self.layers = nn.ModuleList(
            [block_cls(layer_idx, d_model, nhead) for layer_idx in range(num_layers)]
        )
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


def timed_compile_and_run(model, x, label):
    torch._dynamo.reset()
    compiled = torch.compile(model, dynamic=True)

    t0 = time.perf_counter()
    compiled(x)
    torch.cuda.synchronize()
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled(x)
    torch.cuda.synchronize()
    run_time = time.perf_counter() - t0

    print(f"[{label}]")
    print(f"  First call (compile + run): {compile_time:.2f}s")
    print(f"  Second call (run only):     {run_time:.4f}s")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(42)
    x = torch.randn(args.batch, args.seq_len, args.d_model, device=device)

    print(
        f"Model: {args.num_layers} heterogeneous layers, "
        f"d_model={args.d_model}, nhead={args.nhead}"
    )
    print(f"Input: batch={args.batch}, seq_len={args.seq_len}, dynamic=True")
    print(f"Device: {device}")
    print("Layer configs:")
    for layer_idx in range(args.num_layers):
        num_experts, top_k, expansion = layer_config(layer_idx)
        print(
            f"  layer {layer_idx}: num_experts={num_experts}, "
            f"top_k={top_k}, expansion={expansion}"
        )
    print(
        "Tip: run with TORCH_LOGS='+hierarchical_compile,recompiles' "
        "to confirm whether invoke_subgraph was entered."
    )
    print()

    model = HeterogeneousStackedModel(
        HeterogeneousBlockWithRegion, args.num_layers, args.d_model, args.nhead
    ).to(device)
    timed_compile_and_run(
        model,
        x,
        f"With nested_compile_region ({args.num_layers} heterogeneous layers)",
    )


if __name__ == "__main__":
    main()
