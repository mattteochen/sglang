"""
Repro for the whole-model DeepSeek-style loop:

- The top-level model is torch.compile'd.
- It loops over registered layer modules.
- For each layer, it resolves either the raw per-layer implementation or a
  nested_compile_region wrapper based on layer_id.

This is meant to answer whether using nn.ModuleList is what changes the
hierarchical_compile behavior compared with simpler stacked-model repros.

Suggested runs:
  TORCH_LOGS='+hierarchical_compile,recompiles' python test3.py --storage modulelist
  TORCH_LOGS='+hierarchical_compile,recompiles' python test3.py --storage attrs
  TORCH_LOGS='+hierarchical_compile,recompiles' python test3.py --storage both
"""

import argparse
import functools
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMoELayer(nn.Module):
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


class DeepSeekLoopBlock(nn.Module):
    def __init__(self, layer_id, d_model, nhead, num_experts=4, top_k=2):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = SimpleMoELayer(d_model, num_experts=num_experts, top_k=top_k)

    def _forward_prefill_impl(self, x):
        batch, seq_len, dim = x.shape
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        flat = self.ln2(x).view(batch * seq_len, dim)
        x = x + self.moe(flat).view(batch, seq_len, dim)
        return x

    def forward(self, x):
        return self._forward_prefill_impl(x)


@functools.lru_cache(maxsize=None)
def _get_nested_compile_region_fn(fn):
    return torch.compiler.nested_compile_region(
        fn,
        max_reuse_entries=1,
    )


def _should_use_nested_compile_region_for_layer(layer_id, first_wrapped_layer):
    return layer_id >= first_wrapped_layer


def _get_nested_compile_region_target(layer, first_wrapped_layer):
    wrapped_fn = getattr(layer._forward_prefill_impl, "__func__", None)
    if not _should_use_nested_compile_region_for_layer(
        getattr(layer, "layer_id", None), first_wrapped_layer
    ) or wrapped_fn is None:
        return layer._forward_prefill_impl
    return _get_nested_compile_region_fn(wrapped_fn).__get__(layer, type(layer))


def _format_target_resolution(layer, first_wrapped_layer):
    target = _get_nested_compile_region_target(layer, first_wrapped_layer)
    resolved_fn = getattr(target, "__func__", target)
    marked_fn = getattr(resolved_fn, "__marked_compile_region_fn__", None)
    if marked_fn is None:
        return f"{layer.layer_id}:raw"
    return (
        f"{layer.layer_id}:wrapped("
        f"marked_fn={getattr(marked_fn, '__qualname__', type(marked_fn).__name__)}, "
        f"max_reuse_entries={getattr(marked_fn, '__marked_compile_region_max_reuse_entries__', None)})"
    )


class WholeModelLoopModuleList(nn.Module):
    def __init__(self, num_layers, d_model, nhead, first_wrapped_layer):
        super().__init__()
        self.first_wrapped_layer = first_wrapped_layer
        self.start_layer = 0
        self.end_layer = num_layers
        self.layers = nn.ModuleList(
            [DeepSeekLoopBlock(i, d_model, nhead) for i in range(num_layers)]
        )
        self.out = nn.Linear(d_model, d_model, bias=False)

    def target_resolution(self):
        return [
            _format_target_resolution(self.layers[i], self.first_wrapped_layer)
            for i in range(self.start_layer, self.end_layer)
        ]

    def forward(self, x):
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            x = _get_nested_compile_region_target(layer, self.first_wrapped_layer)(x)
        return self.out(x)


class WholeModelLoopAttrs(nn.Module):
    def __init__(self, num_layers, d_model, nhead, first_wrapped_layer):
        super().__init__()
        self.first_wrapped_layer = first_wrapped_layer
        self.start_layer = 0
        self.end_layer = num_layers
        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, f"layer_{i}", DeepSeekLoopBlock(i, d_model, nhead))
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _layer_at(self, idx):
        return getattr(self, f"layer_{idx}")

    def target_resolution(self):
        return [
            _format_target_resolution(self._layer_at(i), self.first_wrapped_layer)
            for i in range(self.start_layer, self.end_layer)
        ]

    def forward(self, x):
        for i in range(self.start_layer, self.end_layer):
            layer = self._layer_at(i)
            x = _get_nested_compile_region_target(layer, self.first_wrapped_layer)(x)
        return self.out(x)


def timed_compile_and_run(model, x, label):
    torch._dynamo.reset()
    compiled = torch.compile(model, dynamic=True)

    print(f"[{label}]")
    print("  Target resolution:")
    for chunk_start in range(0, len(model.target_resolution()), 4):
        chunk = model.target_resolution()[chunk_start : chunk_start + 4]
        print(f"    {', '.join(chunk)}")

    t0 = time.perf_counter()
    compiled(x)
    torch.cuda.synchronize()
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled(x)
    torch.cuda.synchronize()
    run_time = time.perf_counter() - t0

    print(f"  First call (compile + run): {compile_time:.2f}s")
    print(f"  Second call (run only):     {run_time:.4f}s")
    print()


def run_storage(storage, num_layers, d_model, nhead, batch, seq_len, first_wrapped_layer):
    device = "cuda"
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model, device=device)

    if storage == "modulelist":
        model = WholeModelLoopModuleList(
            num_layers, d_model, nhead, first_wrapped_layer
        ).to(device)
        label = f"Whole-model loop with ModuleList ({num_layers} layers)"
    elif storage == "attrs":
        model = WholeModelLoopAttrs(num_layers, d_model, nhead, first_wrapped_layer).to(
            device
        )
        label = f"Whole-model loop with named attrs ({num_layers} layers)"
    else:
        raise ValueError(f"Unsupported storage: {storage}")

    timed_compile_and_run(model, x, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage",
        choices=["modulelist", "attrs", "both"],
        default="modulelist",
        help="Storage style for layers. 'modulelist' mirrors DeepSeek more closely.",
    )
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--first-wrapped-layer", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    args = parser.parse_args()

    print(
        f"Whole-model style loop repro: num_layers={args.num_layers}, "
        f"first_wrapped_layer={args.first_wrapped_layer}, storage={args.storage}"
    )
    print(
        "Tip: run with TORCH_LOGS='+hierarchical_compile,recompiles' to see "
        "whether invoke_subgraph is entered."
    )
    print()

    if args.storage == "both":
        run_storage(
            "modulelist",
            args.num_layers,
            args.d_model,
            args.nhead,
            args.batch,
            args.seq_len,
            args.first_wrapped_layer,
        )
        run_storage(
            "attrs",
            args.num_layers,
            args.d_model,
            args.nhead,
            args.batch,
            args.seq_len,
            args.first_wrapped_layer,
        )
    else:
        run_storage(
            args.storage,
            args.num_layers,
            args.d_model,
            args.nhead,
            args.batch,
            args.seq_len,
            args.first_wrapped_layer,
        )


if __name__ == "__main__":
    main()
