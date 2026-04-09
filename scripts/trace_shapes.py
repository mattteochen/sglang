#!/usr/bin/env python3
"""
Trace all parameter and intermediate tensor shapes of an SGLang model
WITHOUT loading real weights or requiring full GPU memory.

Uses meta device (0 bytes GPU RAM) + gloo backend (no NCCL needed).

Usage — pass the same flags you'd give launch_server (minus --port):

    python scripts/trace_shapes.py \
        --model-path "$MODEL_PATH" \
        --trust-remote-code \
        --kv-cache-dtype fp8_e4m3 \
        --quantization fp8

Notes:
    * --tensor-parallel-size is forced to 1 so shapes reflect a single
      shard.  Multiply sharded dimensions by your real TP size to get
      the un-sharded shape.
    * No weights are downloaded — only the HF config.json is fetched.
    * You can pipe the output to a file:
          python scripts/trace_shapes.py ... > shapes.txt
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Suppress noisy logs from third-party libraries during model registry scan
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
for _quiet in (
    "httpx", "httpcore", "huggingface_hub", "transformers",
    "sglang.srt.server_args", "sglang.srt.configs",
):
    logging.getLogger(_quiet).setLevel(logging.WARNING)


def _sizeof_fmt(num_params: int, bytes_per_param: float) -> str:
    size_gb = num_params * bytes_per_param / (1024**3)
    return f"{size_gb:.2f} GB"


def print_header(title: str):
    logger.info(f"\n{'=' * 80}")
    logger.info(f"  {title}")
    logger.info(f"{'=' * 80}")


def print_config_dimensions(hf_cfg):
    """Print all interesting numeric config attributes."""
    print_header("HF Config — Key Dimensions")

    groups = OrderedDict(
        {
            "General": [
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "max_position_embeddings",
            ],
            "MLA / DeepSeek": [
                "qk_nope_head_dim",
                "qk_rope_head_dim",
                "kv_lora_rank",
                "q_lora_rank",
                "v_head_dim",
                "rope_head_dim",
            ],
            "NSA (Native Sparse Attention)": [
                "index_topk",
                "index_n_heads",
                "index_head_dim",
            ],
            "MoE": [
                "n_routed_experts",
                "num_experts",
                "num_experts_per_tok",
                "n_shared_experts",
                "moe_intermediate_size",
                "first_k_dense_replace",
                "routed_scaling_factor",
            ],
            "Rope": [
                "rope_theta",
                "rope_scaling",
            ],
        }
    )

    for group_name, attrs in groups.items():
        found = []
        for attr in attrs:
            val = getattr(hf_cfg, attr, None)
            if val is not None:
                found.append((attr, val))
        if found:
            logger.info(f"\n  [{group_name}]")
            for attr, val in found:
                logger.info(f"    {attr:40s} = {val}")


def _layer_signature(layer_key: str, params):
    """Fingerprint for a layer: sorted list of (short_name, shape, dtype)."""
    return tuple(
        (
            name[len(layer_key) + 1:] if name.startswith(layer_key) else name,
            tuple(param.shape),
            param.dtype,
        )
        for name, param in params
    )


def print_parameter_shapes(model: nn.Module):
    """Print every named parameter with its shape and dtype, collapsing
    consecutive layers that share the same structure."""
    print_header("All Parameter Shapes (identical consecutive layers collapsed)")

    total_params = 0
    total_elements = 0
    layer_params = OrderedDict()

    for name, param in model.named_parameters():
        total_params += 1
        total_elements += param.numel()

        parts = name.split(".")
        if len(parts) >= 3 and parts[1] == "layers":
            layer_key = f"{parts[0]}.{parts[1]}.{parts[2]}"
        else:
            layer_key = parts[0] if parts else name

        if layer_key not in layer_params:
            layer_params[layer_key] = []
        layer_params[layer_key].append((name, param))

    prev_sig = None
    run_start = None
    run_end = None

    def _flush_run(start_key, end_key, sig, params):
        if start_key == end_key:
            logger.info(f"\n  --- {start_key} ---")
        else:
            start_idx = start_key.rsplit(".", 1)[-1]
            end_idx = end_key.rsplit(".", 1)[-1]
            logger.info(f"\n  --- {start_key} ... {end_key}  (layers {start_idx}-{end_idx}, same structure) ---")
        for short_name, shape, dtype in sig:
            logger.info(
                f"    {short_name:55s}  {str(list(shape)):30s}  {dtype}"
            )

    items = list(layer_params.items())
    for layer_key, params in items:
        sig = _layer_signature(layer_key, params)
        if sig == prev_sig:
            run_end = layer_key
        else:
            if prev_sig is not None:
                _flush_run(run_start, run_end, prev_sig, None)
            run_start = layer_key
            run_end = layer_key
            prev_sig = sig

    if prev_sig is not None:
        _flush_run(run_start, run_end, prev_sig, None)

    logger.info(f"\n  Total parameters:  {total_params:,}")
    logger.info(f"  Total elements:    {total_elements:,}")
    logger.info(f"  Estimated bf16:    {_sizeof_fmt(total_elements, 2)}")
    logger.info(f"  Estimated fp8:     {_sizeof_fmt(total_elements, 1)}")


def print_module_tree(model: nn.Module):
    """Print unique module types with their direct parameters."""
    print_header("Unique Module Types (first occurrence)")

    seen = set()
    for name, module in model.named_modules():
        type_name = type(module).__name__
        if type_name in seen:
            continue
        seen.add(type_name)

        direct_params = {
            pn: list(p.shape)
            for pn, p in module.named_parameters(recurse=False)
        }
        direct_buffers = {
            bn: list(b.shape)
            for bn, b in module.named_buffers(recurse=False)
        }

        logger.info(f"\n  {type_name}  (e.g. '{name}')")
        if direct_params:
            for pn, shape in direct_params.items():
                logger.info(f"    param  .{pn:40s}  {shape}")
        if direct_buffers:
            for bn, shape in direct_buffers.items():
                logger.info(f"    buffer .{bn:40s}  {shape}")
        if not direct_params and not direct_buffers:
            logger.info(f"    (no direct parameters or buffers)")


def print_intermediate_shapes(hf_cfg):
    """Compute theoretical intermediate tensor shapes from config."""
    print_header("Theoretical Intermediate Tensor Shapes (batch=B, seq=S)")

    hidden = getattr(hf_cfg, "hidden_size", None)
    n_heads = getattr(hf_cfg, "num_attention_heads", None)
    n_kv_heads = getattr(hf_cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(hf_cfg, "head_dim", None)
    inter = getattr(hf_cfg, "intermediate_size", None)
    vocab = getattr(hf_cfg, "vocab_size", None)
    kv_lora_rank = getattr(hf_cfg, "kv_lora_rank", None)
    q_lora_rank = getattr(hf_cfg, "q_lora_rank", None)
    qk_nope_head_dim = getattr(hf_cfg, "qk_nope_head_dim", None)
    qk_rope_head_dim = getattr(hf_cfg, "qk_rope_head_dim", None)
    v_head_dim = getattr(hf_cfg, "v_head_dim", None)
    index_topk = getattr(hf_cfg, "index_topk", None)
    index_n_heads = getattr(hf_cfg, "index_n_heads", None)
    index_head_dim = getattr(hf_cfg, "index_head_dim", None)
    moe_inter = getattr(hf_cfg, "moe_intermediate_size", None)

    if head_dim is None and hidden and n_heads:
        head_dim = hidden // n_heads

    shapes = []
    shapes.append(("input_ids", f"[B, S]"))
    if hidden:
        shapes.append(("embed_tokens output", f"[B, S, {hidden}]"))
        shapes.append(("hidden_states (per layer)", f"[B, S, {hidden}]"))
        shapes.append(("residual", f"[B, S, {hidden}]"))
        shapes.append(("post-RMSNorm", f"[B, S, {hidden}]"))

    if kv_lora_rank and q_lora_rank:
        shapes.append(("", ""))
        shapes.append(("--- MLA Attention (DeepSeek) ---", ""))
        shapes.append(("q_a (compressed Q)", f"[B, S, {q_lora_rank}]"))
        if n_heads and qk_nope_head_dim and qk_rope_head_dim:
            full_q_dim = n_heads * (qk_nope_head_dim + qk_rope_head_dim)
            shapes.append(("q_b (expanded Q)", f"[B, S, {full_q_dim}]  = {n_heads} * ({qk_nope_head_dim} + {qk_rope_head_dim})"))
        shapes.append(("kv_a (compressed KV)", f"[B, S, {kv_lora_rank + qk_rope_head_dim}]  = {kv_lora_rank} + {qk_rope_head_dim}"))
        if n_kv_heads and qk_nope_head_dim and v_head_dim:
            full_kv_dim = n_kv_heads * (qk_nope_head_dim + v_head_dim)
            shapes.append(("kv_b (expanded KV)", f"[B, S, {full_kv_dim}]  = {n_kv_heads} * ({qk_nope_head_dim} + {v_head_dim})"))
        if v_head_dim and n_heads:
            shapes.append(("attn_output (pre-proj)", f"[B, S, {n_heads * v_head_dim}]  = {n_heads} * {v_head_dim}"))
    elif n_heads and head_dim:
        shapes.append(("", ""))
        shapes.append(("--- Standard Attention ---", ""))
        shapes.append(("Q", f"[B, S, {n_heads}, {head_dim}]"))
        shapes.append(("K", f"[B, S, {n_kv_heads}, {head_dim}]"))
        shapes.append(("V", f"[B, S, {n_kv_heads}, {head_dim}]"))
        shapes.append(("attn_output", f"[B, S, {n_heads * head_dim}]"))

    if index_topk:
        shapes.append(("", ""))
        shapes.append(("--- NSA Indexer ---", ""))
        if index_n_heads and index_head_dim:
            shapes.append(("index_q", f"[B, S, {index_n_heads}, {index_head_dim}]"))
            shapes.append(("index_k", f"[B, S, {index_n_heads}, {index_head_dim}]"))
        shapes.append(("topk_indices", f"[num_tokens, {index_topk}]"))
        shapes.append(("topk_weights", f"[num_tokens, {index_topk}]"))

    if inter and hidden:
        shapes.append(("", ""))
        shapes.append(("--- MLP ---", ""))
        shapes.append(("gate_up_proj output", f"[B, S, {2 * inter}]  (gate + up fused)"))
        shapes.append(("after SiLU * up", f"[B, S, {inter}]"))
        shapes.append(("down_proj output", f"[B, S, {hidden}]"))

    if moe_inter and hidden:
        n_experts = getattr(hf_cfg, "n_routed_experts", None) or getattr(hf_cfg, "num_experts", None)
        topk = getattr(hf_cfg, "num_experts_per_tok", None)
        n_shared = getattr(hf_cfg, "n_shared_experts", None)
        shapes.append(("", ""))
        shapes.append(("--- MoE MLP ---", ""))
        if topk:
            shapes.append(("router_logits", f"[B*S, {n_experts}]"))
            shapes.append(("topk_weights", f"[B*S, {topk}]"))
            shapes.append(("topk_ids", f"[B*S, {topk}]"))
        shapes.append(("per-expert gate_up", f"[tokens_for_expert, {2 * moe_inter}]"))
        shapes.append(("per-expert down", f"[tokens_for_expert, {hidden}]"))
        if n_shared:
            shared_inter = getattr(hf_cfg, "intermediate_size", inter) or moe_inter
            shapes.append(("shared_expert gate_up", f"[B, S, {2 * shared_inter}]"))
            shapes.append(("shared_expert down", f"[B, S, {hidden}]"))

    if vocab and hidden:
        shapes.append(("", ""))
        shapes.append(("--- Output ---", ""))
        shapes.append(("lm_head output (logits)", f"[B, S, {vocab}]"))

    max_name = max((len(n) for n, _ in shapes if n), default=40)
    for name, shape in shapes:
        if not name:
            logger.info("")
        elif name.startswith("---"):
            logger.info(f"  {name}")
        else:
            logger.info(f"    {name:{max_name}s}  {shape}")


def main():
    from sglang.srt.server_args import (
        ServerArgs,
        prepare_server_args,
        set_global_server_args_for_scheduler,
    )

    argv = sys.argv[1:]
    server_args = prepare_server_args(argv)

    original_tp = server_args.tp_size
    if server_args.tp_size > 1:
        logger.info(
            f"[info] Overriding --tensor-parallel-size {server_args.tp_size} -> 1 "
            f"for local shape tracing.  Sharded dims will reflect TP=1; "
            f"multiply by {original_tp} for the un-sharded shape."
        )
        server_args.tp_size = 1
    server_args.pp_size = 1
    server_args.dp_size = 1
    server_args.ep_size = 1
    server_args.attn_cp_size = 1
    server_args.moe_dp_size = 1

    # Initialize minimal distributed state (gloo, single process)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
        backend="gloo",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    set_global_server_args_for_scheduler(server_args)

    from sglang.srt.configs.load_config import LoadConfig, LoadFormat
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
    from sglang.srt.model_loader.loader import _initialize_model
    from sglang.srt.model_loader.utils import set_default_torch_dtype

    model_config = ModelConfig.from_server_args(server_args)

    hf_cfg = model_config.hf_config
    arch = getattr(hf_cfg, "architectures", ["unknown"])

    print_header(f"Model Shape Trace")
    logger.info(f"  model_path:    {model_config.model_path}")
    logger.info(f"  architecture:  {arch}")
    logger.info(f"  dtype:         {model_config.dtype}")
    logger.info(f"  original TP:   {original_tp}")

    # ---- 1. Config dimensions ----
    print_config_dimensions(hf_cfg)

    # ---- 2. Theoretical intermediate shapes ----
    print_intermediate_shapes(hf_cfg)

    # ---- 3. Instantiate on meta device ----
    load_config = LoadConfig(load_format=LoadFormat.DUMMY)

    from sglang.srt.model_loader.utils import get_model_architecture

    logger.info("\n[info] Resolving model architecture...")
    monkey_patch_vllm_parallel_state()
    logging.getLogger("sglang.srt.models.registry").setLevel(logging.ERROR)
    model_class, resolved_arch = get_model_architecture(model_config)
    logging.getLogger("sglang.srt.models.registry").setLevel(logging.INFO)
    logger.info(f"  resolved class: {model_class.__name__} ({resolved_arch})")

    logger.info("[info] Instantiating model on meta device (0 GPU memory)...")
    try:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = model_class(config=model_config.hf_config, quant_config=None)
    except Exception as e:
        logger.error(f"\n[error] Model instantiation failed: {e}")
        logger.error(
            "[hint] Some models have init-time ops that don't work on meta device.\n"
            "       The config dimensions and theoretical shapes above are still valid."
        )
        return
    finally:
        monkey_patch_vllm_parallel_state(reverse=True)

    # ---- 4. Parameter shapes ----
    print_parameter_shapes(model)

    # ---- 5. Module tree ----
    print_module_tree(model)

    logger.info(f"\n{'=' * 80}")
    logger.info("  Done. All shapes above reflect TP=1.")
    if original_tp > 1:
        logger.info(
            f"  Your server uses TP={original_tp}.  Tensor-parallel-sharded "
            f"dimensions\n  (e.g. num_heads, hidden projections) are divided "
            f"by {original_tp} at runtime."
        )
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
