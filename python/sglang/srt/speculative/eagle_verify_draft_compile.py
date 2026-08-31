"""Opt-in Inductor island between EAGLE target verify and draft extend.

The serving objects and CUDA stream/event orchestration stay eager.  This module
only flattens their tensor work into one full graph.  External sampling and NCCL
operations are explicit opaque mutation barriers inside that graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.distributed.parallel_state import inplace_broadcast
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)
_activation_logged = False


@register_custom_op(
    op_name="compiled_tree_speculative_sampling_target_only",
    mutates_args=["predicts", "accept_index", "accept_token_num", "draft_probs"],
)
def _compiled_tree_speculative_sampling_target_only(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float,
    threshold_acc: float,
) -> None:
    # The underlying sgl-kernel schema omits draft_probs from its mutation list,
    # although the CUDA implementation writes it.  Keep that implementation
    # behind a schema that tells FunctionalTensor/FakeTensor the full truth.
    from sgl_kernel import tree_speculative_sampling_target_only

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        retrive_next_token=retrieve_next_token,
        retrive_next_sibling=retrieve_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )


@dataclass
class EagleVerifyDraftTensorPlan:
    predict: torch.Tensor
    accept_lens: torch.Tensor
    accept_index: torch.Tensor
    new_seq_lens: torch.Tensor
    bonus_tokens: torch.Tensor
    last_correct_step_indices: torch.Tensor
    mamba_steps_to_track: Optional[torch.Tensor]
    draft_select_index: torch.Tensor
    next_token_ids_i64: torch.Tensor
    draft_prefix_lens: torch.Tensor
    draft_extend_lens: torch.Tensor
    draft_seq_lens: torch.Tensor


@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"triton.cudagraphs": False},
)
def _compiled_verify_draft_tensor_graph(
    next_token_logits: torch.Tensor,
    temperatures: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    coins: torch.Tensor,
    coins_for_final_sampling: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    max_tree_depth: int,
    threshold_single: float,
    threshold_acc: float,
    tp_group_name: str,
    tp_world_size: int,
    mamba_track_interval: int,
    simulated_accept_len: int,
    simulate_real_draft_tokens: bool,
):
    bs = candidates.shape[0]
    expanded_temperature = torch.repeat_interleave(temperatures, draft_token_num, dim=0)
    target_probs = torch.softmax(
        next_token_logits / expanded_temperature, dim=-1
    ).reshape(bs, draft_token_num, -1)
    draft_probs = torch.zeros_like(target_probs)

    predict = torch.zeros(
        next_token_logits.shape[:-1],
        dtype=torch.int32,
        device=next_token_logits.device,
    ).flatten()
    accept_index = torch.full(
        (bs, max_tree_depth),
        -1,
        dtype=torch.int32,
        device=next_token_logits.device,
    )
    num_correct_drafts = torch.empty(
        (bs,), dtype=torch.int32, device=next_token_logits.device
    )
    _compiled_tree_speculative_sampling_target_only(
        predict,
        accept_index,
        num_correct_drafts,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        coins,
        coins_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
    )

    if tp_world_size > 1:
        inplace_broadcast(predict, tp_group_name, 0)
        inplace_broadcast(accept_index, tp_group_name, 0)
        inplace_broadcast(num_correct_drafts, tp_group_name, 0)

    if simulated_accept_len > 0:
        # Preserve the benchmark-only simulated-acceptance policy, but let
        # Inductor fuse its many tiny fill/arange/index/update operations.
        target_predict = torch.argmax(next_token_logits, dim=-1).reshape(
            bs, draft_token_num
        )
        simulated_accept_index = torch.full_like(accept_index, -1)
        simulated_accept_index[:, :simulated_accept_len] = accept_index[
            :, :1
        ] + torch.arange(simulated_accept_len, device=accept_index.device)
        num_correct_drafts.fill_(simulated_accept_len - 1)
        if simulate_real_draft_tokens:
            if simulated_accept_len > 1:
                draft_node_indices = simulated_accept_index[
                    :, : simulated_accept_len - 1
                ].to(torch.int64)
                predict[draft_node_indices] = candidates[:, 1:simulated_accept_len].to(
                    predict.dtype
                )
            bonus_node_indices = simulated_accept_index[:, simulated_accept_len - 1].to(
                torch.int64
            )
            predict[bonus_node_indices] = target_predict[
                :, simulated_accept_len - 1
            ].to(predict.dtype)
        else:
            predict.fill_(100)
        accept_index = simulated_accept_index

    accept_lens = num_correct_drafts + 1
    new_seq_lens = seq_lens + accept_lens
    req_idx = torch.arange(bs, dtype=torch.int64, device=seq_lens.device)
    bonus_column = (accept_lens - 1).to(torch.int64)
    bonus_index = accept_index[req_idx, bonus_column]
    bonus_tokens = predict[bonus_index]

    accept_indices_offset = torch.arange(
        0,
        bs * draft_token_num,
        step=draft_token_num,
        dtype=accept_lens.dtype,
        device=accept_lens.device,
    )
    last_correct_step_indices = (
        accept_index[req_idx, bonus_column] - accept_indices_offset
    )
    if mamba_track_interval > 0:
        seq_lens_post_verify = seq_lens + accept_lens
        to_track_mask = (
            seq_lens // mamba_track_interval
            != seq_lens_post_verify // mamba_track_interval
        )
        tracking_point = (
            seq_lens_post_verify // mamba_track_interval * mamba_track_interval
        )
        to_track_ith = torch.clamp(tracking_point - seq_lens - 1, min=0).to(torch.int64)
        candidate_track_steps = (
            accept_index[req_idx, to_track_ith] - accept_indices_offset
        )
        mamba_steps_to_track = torch.where(
            to_track_mask,
            candidate_track_steps,
            torch.full_like(candidate_track_steps, -1),
        )
    else:
        mamba_steps_to_track = torch.empty(
            (0,), dtype=torch.int32, device=seq_lens.device
        )

    draft_select_index = (
        torch.arange(
            0,
            bs * draft_token_num,
            step=draft_token_num,
            device=seq_lens.device,
        )
        + accept_lens
        - 1
    )
    next_token_ids_i64 = predict.to(torch.int64)
    draft_prefix_lens = (seq_lens - 0).clamp(min=0).to(torch.int32)
    draft_extend_lens = torch.full(
        (bs,), draft_token_num, dtype=torch.int32, device=seq_lens.device
    )
    draft_seq_lens = seq_lens + draft_token_num
    return (
        predict,
        accept_lens,
        accept_index,
        new_seq_lens,
        bonus_tokens,
        last_correct_step_indices,
        mamba_steps_to_track,
        draft_select_index,
        next_token_ids_i64,
        draft_prefix_lens,
        draft_extend_lens,
        draft_seq_lens,
    )


def run_compiled_verify_draft_tensor_graph(
    *,
    next_token_logits: torch.Tensor,
    temperatures: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    coins: torch.Tensor,
    coins_for_final_sampling: torch.Tensor,
    seq_lens: torch.Tensor,
    draft_token_num: int,
    max_tree_depth: int,
    threshold_single: float,
    threshold_acc: float,
    tp_group_name: str,
    tp_world_size: int,
    mamba_track_interval: Optional[int],
    simulated_accept_len: int,
    simulate_real_draft_tokens: bool,
) -> EagleVerifyDraftTensorPlan:
    global _activation_logged
    values = _compiled_verify_draft_tensor_graph(
        next_token_logits,
        temperatures,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        coins,
        coins_for_final_sampling,
        seq_lens,
        draft_token_num,
        max_tree_depth,
        threshold_single,
        threshold_acc,
        tp_group_name,
        tp_world_size,
        mamba_track_interval or 0,
        simulated_accept_len,
        simulate_real_draft_tokens,
    )
    if not _activation_logged:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info(
            "MTP_VERIFY_DRAFT_COMPILE_ACTIVE rank=%d tp_world_size=%d "
            "draft_token_num=%d max_tree_depth=%d",
            rank,
            tp_world_size,
            draft_token_num,
            max_tree_depth,
        )
        _activation_logged = True
    return EagleVerifyDraftTensorPlan(
        predict=values[0],
        accept_lens=values[1],
        accept_index=values[2],
        new_seq_lens=values[3],
        bonus_tokens=values[4],
        last_correct_step_indices=values[5],
        mamba_steps_to_track=(values[6] if mamba_track_interval is not None else None),
        draft_select_index=values[7],
        next_token_ids_i64=values[8],
        draft_prefix_lens=values[9],
        draft_extend_lens=values[10],
        draft_seq_lens=values[11],
    )
