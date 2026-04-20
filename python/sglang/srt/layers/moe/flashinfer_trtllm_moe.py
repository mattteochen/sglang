from typing import Optional

import torch

from sglang.srt.utils.common import next_power_of_2
from sglang.srt.utils.custom_op import register_custom_op_from_extern


def _compute_tune_max_num_tokens(
    hidden_states: torch.Tensor, **_: object
) -> int:
    return next_power_of_2(int(hidden_states.shape[0]))


def _register_fp8_trtllm_moe_custom_op(fn, *, op_name: str):
    # Keep tune_max_num_tokens out of the traced op schema. Computing
    # next_power_of_2(hidden_states.shape[0]) in Python forces Dynamo to
    # specialize on the concrete token count and triggers recompilation.
    return register_custom_op_from_extern(
        fn,
        op_name=op_name,
        out_shape="hidden_states",
        out_dtype=torch.bfloat16,
        computed_args={"tune_max_num_tokens": _compute_tune_max_num_tokens},
    )


def _trtllm_fp8_block_scale_moe_impl(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type: Optional[int] = None,
) -> torch.Tensor:
    try:
        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
    except ImportError as e:
        raise ImportError(
            "Can't import trtllm_fp8_block_scale_moe from flashinfer. "
            "Please check flashinfer version."
        ) from e

    kwargs = {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "num_experts": num_experts,
        "top_k": top_k,
        "n_group": n_group,
        "topk_group": topk_group,
        "intermediate_size": intermediate_size,
        "local_expert_offset": local_expert_offset,
        "local_num_experts": local_num_experts,
        "routed_scaling_factor": routed_scaling_factor,
        "routing_method_type": routing_method_type,
        "use_shuffled_weight": use_shuffled_weight,
        "weight_layout": weight_layout,
        "enable_pdl": enable_pdl,
        "tune_max_num_tokens": tune_max_num_tokens,
    }
    if fp8_quantization_type is not None:
        from flashinfer.fused_moe import Fp8QuantizationType

        kwargs["fp8_quantization_type"] = Fp8QuantizationType(fp8_quantization_type)

    return trtllm_fp8_block_scale_moe(**kwargs)


trtllm_fp8_block_scale_moe_wrapper = _register_fp8_trtllm_moe_custom_op(
    _trtllm_fp8_block_scale_moe_impl,
    op_name="trtllm_fp8_block_scale_moe_wrapper",
)


def _trtllm_fp8_block_scale_routed_moe_impl(
    topk_ids: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type: Optional[int] = None,
) -> torch.Tensor:
    try:
        from flashinfer.fused_moe import trtllm_fp8_block_scale_routed_moe
    except ImportError as e:
        raise ImportError(
            "Can't import trtllm_fp8_block_scale_routed_moe from flashinfer. "
            "Please check flashinfer version."
        ) from e

    kwargs = {
        "topk_ids": topk_ids,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
        "num_experts": num_experts,
        "top_k": top_k,
        "n_group": n_group,
        "topk_group": topk_group,
        "intermediate_size": intermediate_size,
        "local_expert_offset": local_expert_offset,
        "local_num_experts": local_num_experts,
        "routed_scaling_factor": routed_scaling_factor,
        "routing_method_type": routing_method_type,
        "use_shuffled_weight": use_shuffled_weight,
        "weight_layout": weight_layout,
        "enable_pdl": enable_pdl,
        "tune_max_num_tokens": tune_max_num_tokens,
    }
    if fp8_quantization_type is not None:
        from flashinfer.fused_moe import Fp8QuantizationType

        kwargs["fp8_quantization_type"] = Fp8QuantizationType(fp8_quantization_type)

    return trtllm_fp8_block_scale_routed_moe(**kwargs)


trtllm_fp8_block_scale_routed_moe_wrapper = _register_fp8_trtllm_moe_custom_op(
    _trtllm_fp8_block_scale_routed_moe_impl,
    op_name="trtllm_fp8_block_scale_routed_moe_wrapper",
)


def _trtllm_fp8_per_tensor_scale_moe_impl(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    try:
        from flashinfer.fused_moe import trtllm_fp8_per_tensor_scale_moe
    except ImportError as e:
        raise ImportError(
            "Can't import trtllm_fp8_per_tensor_scale_moe from flashinfer. "
            "Please check flashinfer version."
        ) from e

    return trtllm_fp8_per_tensor_scale_moe(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        output1_scales_scalar=output1_scales_scalar,
        output1_scales_gate_scalar=output1_scales_gate_scalar,
        gemm2_weights=gemm2_weights,
        output2_scales_scalar=output2_scales_scalar,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        use_routing_scales_on_input=use_routing_scales_on_input,
        routing_method_type=routing_method_type,
        enable_pdl=enable_pdl,
        tune_max_num_tokens=tune_max_num_tokens,
    )


trtllm_fp8_per_tensor_scale_moe_wrapper = _register_fp8_trtllm_moe_custom_op(
    _trtllm_fp8_per_tensor_scale_moe_impl,
    op_name="trtllm_fp8_per_tensor_scale_moe_wrapper",
)
