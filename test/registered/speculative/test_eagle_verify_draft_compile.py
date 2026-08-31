import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sgl_kernel import tree_speculative_sampling_target_only
from sglang.srt.speculative.eagle_verify_draft_compile import (
    _compiled_verify_draft_tensor_graph,
    _functionalize_simulated_acceptance,
)
from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.speculative.spec_utils import generate_simulated_accept_index


class TestEagleVerifyDraftCompile(unittest.TestCase):
    def test_eagle_sample_routes_real_c2_simulation_to_compiled_plan(self):
        import sglang.srt.speculative.eagle_utils as eagle_utils

        plan = SimpleNamespace(
            predict=torch.arange(5, dtype=torch.int32),
            num_correct_drafts=torch.tensor([2], dtype=torch.int32),
            accept_lens=torch.tensor([3], dtype=torch.int32),
            accept_index=torch.tensor([[0, 1, 2, -1, -1]], dtype=torch.int32),
        )
        sampling_info = SimpleNamespace(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            is_all_greedy=False,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            temperatures=torch.ones((1, 1)),
        )
        batch = SimpleNamespace(
            device=torch.device("cpu"),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            seq_lens=torch.tensor([126], dtype=torch.int64),
            sampling_info=sampling_info,
            mamba_track_indices=None,
        )
        verify_input = SimpleNamespace(
            draft_token=torch.arange(5, dtype=torch.int64),
            draft_token_num=5,
            max_tree_depth=5,
            tree_topk=1,
            retrieve_index=torch.arange(5, dtype=torch.int64).reshape(1, 5),
            retrieve_next_token=torch.tensor([[1, 2, 3, 4, -1]]),
            retrieve_next_sibling=torch.full((1, 5), -1),
        )
        logits_output = SimpleNamespace(next_token_logits=torch.ones((5, 20)))
        spec = SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )
        tp_group = SimpleNamespace(unique_name="tp", world_size=8)

        with (
            mock.patch.object(eagle_utils, "_is_cpu", False),
            mock.patch.object(eagle_utils, "_is_cuda", True),
            mock.patch.object(eagle_utils, "_is_npu", False),
            mock.patch.object(eagle_utils, "_is_hip", False),
            mock.patch.object(eagle_utils, "_is_xpu", False),
            mock.patch.object(eagle_utils, "get_spec", return_value=spec),
            mock.patch.object(
                eagle_utils,
                "get_server_args",
                return_value=SimpleNamespace(enable_multi_layer_eagle=False),
            ),
            mock.patch.object(
                eagle_utils.envs.SGLANG_ENABLE_MTP_VERIFY_DRAFT_COMPILE,
                "get",
                return_value=True,
            ),
            mock.patch.object(
                eagle_utils.envs.SGLANG_ENABLE_MTP_VERIFY_DRAFT_FUNCTIONAL_SIM,
                "get",
                return_value=True,
            ),
            mock.patch.object(
                eagle_utils.envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM,
                "get",
                return_value=False,
            ),
            mock.patch.object(
                eagle_utils, "_verify_coins", return_value=(torch.ones((1, 5)), torch.ones(1))
            ),
            mock.patch(
                "sglang.srt.speculative.spec_utils.SIMULATE_ACC_LEN", 3.33
            ),
            mock.patch(
                "sglang.srt.speculative.spec_utils.SIMULATE_ACC_METHOD",
                "match-expected",
            ),
            mock.patch(
                "sglang.srt.speculative.spec_utils.SIMULATE_ACC_TOKEN_MODE",
                "real-draft-token",
            ),
            mock.patch(
                "sglang.srt.speculative.spec_utils._sample_simulated_acc_len",
                return_value=3,
            ),
            mock.patch(
                "sglang.srt.speculative.eagle_verify_draft_compile.run_compiled_verify_draft_tensor_graph",
                return_value=plan,
            ) as compiled,
            mock.patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
            mock.patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
        ):
            got = eagle_sample(verify_input, batch, logits_output, grammar_mask=None)

        self.assertIs(verify_input._verify_draft_tensor_plan, plan)
        self.assertEqual(compiled.call_args.kwargs["simulated_accept_len"], 3)
        self.assertEqual(compiled.call_args.kwargs["tp_world_size"], 8)
        self.assertTrue(compiled.call_args.kwargs["simulate_real_draft_tokens"])
        self.assertTrue(compiled.call_args.kwargs["functional_simulation"])
        for actual, expected in zip(got, (plan.predict, plan.accept_lens, plan.accept_index)):
            torch.testing.assert_close(actual, expected)

    def _inputs(self, bs: int):
        candidates = torch.tensor(
            [[0, 1, 2, 3, 4], [7, 8, 9, 10, 11]],
            dtype=torch.int64,
            device="cuda",
        )[:bs]
        retrieve_index = torch.tensor(
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            dtype=torch.int64,
            device="cuda",
        )[:bs]
        retrieve_next_token = torch.tensor(
            [[1, 2, 3, 4, -1], [4, 2, 3, -1, -1]],
            dtype=torch.int64,
            device="cuda",
        )[:bs]
        retrieve_next_sibling = torch.tensor(
            [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, 1]],
            dtype=torch.int64,
            device="cuda",
        )[:bs]
        logits = torch.ones((bs, 5, 20), dtype=torch.float32, device="cuda")
        logits[0, 0, 3] = logits[0, 3, 4] = logits[0, 4, 5] = 10
        if bs == 2:
            logits[1, 0, 11] = logits[1, 4, 12] = 10
        for i in range(bs):
            for j in range(5):
                if torch.max(logits[i, j]) < 10:
                    logits[i, j, 18] = 10
        return (
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            logits.reshape(bs * 5, 20),
            torch.full((bs, 1), 0.01, dtype=torch.float32, device="cuda"),
            torch.full((bs, 5), 0.5, dtype=torch.float32, device="cuda"),
            torch.full((bs,), 0.5, dtype=torch.float32, device="cuda"),
            torch.arange(126, 126 + bs, dtype=torch.int64, device="cuda"),
        )

    def _reference(
        self,
        inputs,
        simulated_accept_len,
        mamba_track_interval,
        simulate_real_draft_tokens=True,
    ):
        (
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            logits,
            temperatures,
            coins,
            final_coins,
            seq_lens,
        ) = inputs
        bs, width = candidates.shape
        probs = torch.softmax(
            logits / torch.repeat_interleave(temperatures, width, dim=0), dim=-1
        ).reshape(bs, width, -1)
        predict = torch.zeros((bs * width,), dtype=torch.int32, device="cuda")
        accept_index = torch.full((bs, 5), -1, dtype=torch.int32, device="cuda")
        correct = torch.empty((bs,), dtype=torch.int32, device="cuda")
        draft_probs = torch.zeros_like(probs)
        tree_speculative_sampling_target_only(
            predicts=predict,
            accept_index=accept_index,
            accept_token_num=correct,
            candidates=candidates,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=final_coins,
            target_probs=probs,
            draft_probs=draft_probs,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )
        row_offsets = torch.arange(0, bs * width, width, device="cuda")
        self.assertTrue(torch.all(accept_index[:, 0] >= row_offsets).item())
        self.assertTrue(
            torch.all(
                accept_index[:, 0] + simulated_accept_len <= row_offsets + width
            ).item()
        )
        target_predict = torch.argmax(logits, dim=-1).reshape(bs, width)
        accept_index = generate_simulated_accept_index(
            accept_index,
            predict,
            correct,
            candidates,
            target_predict,
            bs,
            4,
            simulate_acc_len=simulated_accept_len,
            simulate_acc_method="match-expected",
            simulate_acc_token_mode=(
                "real-draft-token" if simulate_real_draft_tokens else "fixed"
            ),
        )
        accept_lens = correct + 1
        req_idx = torch.arange(bs, device="cuda")
        bonus_column = (accept_lens - 1).long()
        bonus = predict[accept_index[req_idx, bonus_column]]
        offsets = torch.arange(
            0, bs * width, width, dtype=accept_lens.dtype, device="cuda"
        )
        last = accept_index[req_idx, bonus_column] - offsets
        if mamba_track_interval:
            new_seq_lens = seq_lens + accept_lens
            to_track = (
                seq_lens // mamba_track_interval
                != new_seq_lens // mamba_track_interval
            )
            tracking_point = (
                new_seq_lens // mamba_track_interval * mamba_track_interval
            )
            track_step = torch.clamp(tracking_point - seq_lens - 1, min=0).long()
            candidate_track = accept_index[req_idx, track_step] - offsets
            mamba_steps = torch.where(
                to_track, candidate_track, torch.full_like(candidate_track, -1)
            )
        else:
            mamba_steps = torch.empty((0,), dtype=torch.int32, device="cuda")
        return (
            predict,
            correct,
            accept_lens,
            accept_index,
            seq_lens + accept_lens,
            bonus,
            last,
            mamba_steps,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_matches_eager_simulated_acceptance_bs1_bs2(self):
        # This exhaustive static-argument matrix intentionally exceeds the
        # serving default; production C2 uses only the bounded 5 variants
        # documented by the profile. Restore the process default afterward.
        old_recompile_limit = torch._dynamo.config.recompile_limit
        torch._dynamo.config.recompile_limit = 64
        self.addCleanup(
            setattr, torch._dynamo.config, "recompile_limit", old_recompile_limit
        )
        # 3.33 with match-expected produces both specializations in C2.
        for functional_simulation in (False, True):
            for bs, simulated_accept_len, mamba_track_interval, real_tokens, seqs in (
                (1, 3, 0, True, None),
                (2, 4, 128, True, (120, 127)),
                (1, 4, 128, True, None),
                (2, 3, 128, True, (120, 127)),
                (1, 3, 0, False, None),
                (1, 1, 0, True, None),
                (2, 5, 128, True, (120, 127)),
                (2, 5, 0, False, None),
                (2, 3, 0, True, (-2, 0)),
            ):
                inputs = self._inputs(bs)
                if seqs is not None:
                    inputs[8].copy_(torch.tensor(seqs, device="cuda"))
                actual = _compiled_verify_draft_tensor_graph(
                    inputs[4],
                    inputs[5],
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[6],
                    inputs[7],
                    inputs[8],
                    5,
                    5,
                    1.0,
                    1.0,
                    "",
                    1,
                    mamba_track_interval,
                    simulated_accept_len,
                    real_tokens,
                    functional_simulation,
                )
                expected = self._reference(
                    inputs,
                    simulated_accept_len,
                    mamba_track_interval,
                    real_tokens,
                )
                for got, want in zip(actual[:8], expected):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual[8],
                    expected[2].long()
                    - 1
                    + torch.arange(bs, device="cuda") * 5,
                )
                torch.testing.assert_close(actual[9], expected[0].long())
                torch.testing.assert_close(actual[10], inputs[8].clamp(min=0).int())
                torch.testing.assert_close(
                    actual[11],
                    torch.full((bs,), 5, dtype=torch.int32, device="cuda"),
                )
                torch.testing.assert_close(actual[12], inputs[8] + 5)
                expected_positions = (
                    inputs[8].clamp(min=0).unsqueeze(1)
                    + torch.arange(5, dtype=torch.int64, device="cuda").unsqueeze(0)
                ).reshape(-1)
                torch.testing.assert_close(actual[13], expected_positions)
                torch.testing.assert_close(
                    actual[14],
                    torch.arange(bs, dtype=torch.int32, device="cuda") * 5,
                )
                self.assertTrue(actual[13].is_contiguous())
                self.assertTrue(actual[14].is_contiguous())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_functional_simulation_preserves_arbitrary_off_path_values(self):
        for bs in (1, 2):
            candidates = torch.arange(
                bs * 5, dtype=torch.int64, device="cuda"
            ).reshape(bs, 5)
            target_predict = candidates + 100
            sampler_predict = torch.arange(
                1000, 1000 + bs * 5, dtype=torch.int32, device="cuda"
            )
            sampler_accept_index = torch.full(
                (bs, 5), -1, dtype=torch.int32, device="cuda"
            )
            sampler_accept_index[:, 0] = torch.arange(
                0, bs * 5, 5, dtype=torch.int32, device="cuda"
            )
            torch.testing.assert_close(
                sampler_accept_index[:, 0],
                torch.arange(0, bs * 5, 5, dtype=torch.int32, device="cuda"),
            )
            for simulated_accept_len in (1, 3, 4, 5):
                for real_tokens in (False, True):
                    expected_predict = sampler_predict.clone()
                    expected_correct = torch.empty(
                        (bs,), dtype=torch.int32, device="cuda"
                    )
                    expected_accept_index = generate_simulated_accept_index(
                        accept_index=sampler_accept_index,
                        predict=expected_predict,
                        num_correct_drafts=expected_correct,
                        candidates=candidates,
                        target_predict=target_predict,
                        simulate_acc_len=simulated_accept_len,
                        simulate_acc_method="match-expected",
                        simulate_acc_token_mode=(
                            "real-draft-token" if real_tokens else "fixed"
                        ),
                        bs=bs,
                        spec_steps=4,
                    )
                    actual = _functionalize_simulated_acceptance(
                        sampler_predict,
                        sampler_accept_index,
                        candidates,
                        target_predict,
                        simulated_accept_len,
                        5,
                        real_tokens,
                    )
                    torch.testing.assert_close(actual[0], expected_predict)
                    torch.testing.assert_close(actual[1], expected_correct)
                    torch.testing.assert_close(
                        actual[2], expected_correct + 1
                    )
                    torch.testing.assert_close(actual[3], expected_accept_index)
                    self.assertEqual(actual[0].dtype, torch.int32)
                    self.assertEqual(actual[3].dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
