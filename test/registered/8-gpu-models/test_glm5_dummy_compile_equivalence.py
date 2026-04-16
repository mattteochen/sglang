import os
import subprocess
import sys
import tempfile
import unittest

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, suite="stage-c-test-8-gpu-h200")

GLM5_MODEL_PATH = "zai-org/GLM-5-FP8"
INPUT_LEN = 1024
OUTPUT_LEN = 1024
RANDOM_SEED = 1234
ATOL = 1e-2
RTOL = 1e-2
TIMEOUT_SECONDS = 1800


def _get_gpu_count() -> int:
    if hasattr(torch, "accelerator"):
        try:
            return torch.accelerator.device_count()
        except Exception:
            return 0
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def _build_bench_one_batch_command(logits_path: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.bench_one_batch",
        "--model-path",
        GLM5_MODEL_PATH,
        "--trust-remote-code",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--tensor-parallel-size",
        "8",
        "--mem-fraction-static",
        "0.8",
        "--enable-flashinfer-allreduce-fusion",
        "--quantization",
        "fp8",
        "--nsa-decode-backend",
        "trtllm",
        "--nsa-prefill-backend",
        "trtllm",
        "--moe-runner-backend",
        "flashinfer_trtllm",
        "--load-format",
        "dummy",
        "--batch-size",
        "1",
        "--input-len",
        str(INPUT_LEN),
        "--output-len",
        str(OUTPUT_LEN),
        "--cuda-graph-bs",
        "1",
        "--json-model-override-args",
        '{"num_hidden_layers": 8}',
        "--result-filename",
        "",
        "--random-seed",
        str(RANDOM_SEED),
        "--save-prefill-logits-filename",
        logits_path,
    ]


class TestGLM5DummyCompileEquivalence(unittest.TestCase):
    @unittest.skipIf(_get_gpu_count() < 8, "Requires 8 GPUs")
    def test_glm5_dummy_prefill_logits_match_compiled_and_eager(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            eager_logits_path = os.path.join(tmpdir, "eager_prefill_logits.pt")
            compiled_logits_path = os.path.join(tmpdir, "compiled_prefill_logits.pt")

            eager_payload, eager_output = self._run_bench_one_batch(
                eager_logits_path,
                env_overrides={
                    "SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_MODEL": "0",
                    "SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_SKIP_GUARD_EVAL_UNSAFE": "0",
                    "SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_NESTED_COMPILE_REGION": "0",
                },
            )
            compiled_payload, compiled_output = self._run_bench_one_batch(
                compiled_logits_path,
                env_overrides={
                    "SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_MODEL": "1",
                    "SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_SKIP_GUARD_EVAL_UNSAFE": "1",
                    "SGLANG_EXPERIMENTAL_COMPILE_DEEPSEEK_PREFILL_NESTED_COMPILE_REGION": "0",
                },
            )

            self.assertEqual(
                eager_payload["input_ids"],
                compiled_payload["input_ids"],
                msg="Synthetic inputs differed between eager and compiled runs.",
            )

            eager_logits = eager_payload["next_token_logits"].to(torch.float32)
            compiled_logits = compiled_payload["next_token_logits"].to(torch.float32)
            self.assertEqual(eager_logits.shape, compiled_logits.shape)

            max_abs_diff = (compiled_logits - eager_logits).abs().max().item()
            max_rel_diff = (
                (compiled_logits - eager_logits).abs()
                / eager_logits.abs().clamp_min(1e-6)
            ).max().item()

            try:
                torch.testing.assert_close(
                    compiled_logits,
                    eager_logits,
                    atol=ATOL,
                    rtol=RTOL,
                )
            except AssertionError as exc:
                raise AssertionError(
                    "GLM-5 dummy-weight prefill logits mismatch between compiled "
                    f"and eager runs with max_abs_diff={max_abs_diff:.6f}, "
                    f"max_rel_diff={max_rel_diff:.6f}.\n"
                    f"Eager output:\n{eager_output}\n\n"
                    f"Compiled output:\n{compiled_output}"
                ) from exc

    def _run_bench_one_batch(
        self, logits_path: str, *, env_overrides: dict[str, str]
    ) -> tuple[dict, str]:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"/opt/sglang/python:{existing_pythonpath}"
            if existing_pythonpath
            else "/opt/sglang/python"
        )
        env.update(env_overrides)

        process = subprocess.Popen(
            _build_bench_one_batch_command(logits_path),
            cwd="/opt/sglang",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = ""
        try:
            output, _ = process.communicate(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            kill_process_tree(process.pid)
            raise AssertionError(
                "bench_one_batch timed out while producing prefill logits."
            ) from exc
        finally:
            kill_process_tree(process.pid)

        if process.returncode != 0:
            raise AssertionError(
                f"bench_one_batch exited with code {process.returncode}.\n{output}"
            )
        if not os.path.exists(logits_path):
            raise AssertionError(
                "bench_one_batch did not write the requested prefill logits file.\n"
                f"{output}"
            )

        return torch.load(logits_path, map_location="cpu", weights_only=False), output


if __name__ == "__main__":
    unittest.main()
