import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.http_server import (
    _build_startup_warmup_generate_payloads,
    _get_explicit_startup_warmup_payloads,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class TestHttpServerWarmup(unittest.TestCase):
    def test_build_generate_payloads_for_single_dp(self):
        server_args = SimpleNamespace(dp_size=1)

        payloads = _build_startup_warmup_generate_payloads(server_args, [4, 7])

        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0]["input_ids"], [10, 11, 12, 10])
        self.assertEqual(payloads[1]["input_ids"], [10, 11, 12, 10, 11, 12, 10])
        self.assertEqual(payloads[0]["sampling_params"]["max_new_tokens"], 0)

    def test_build_generate_payloads_for_multi_dp(self):
        server_args = SimpleNamespace(dp_size=2)

        payloads = _build_startup_warmup_generate_payloads(server_args, [3])

        self.assertEqual(
            payloads[0]["input_ids"],
            [
                [10, 11, 12],
                [10, 11, 12],
            ],
        )

    def test_get_explicit_payloads_supported_for_generate(self):
        server_args = SimpleNamespace(
            warmup_input_lens=[8],
            debug_tensor_dump_input_file=None,
            disaggregation_mode="null",
            dp_size=1,
        )

        payloads = _get_explicit_startup_warmup_payloads(server_args, "/generate")

        self.assertIsNotNone(payloads)
        self.assertEqual(payloads[0]["input_ids"], [10, 11, 12, 10, 11, 12, 10, 11])

    def test_get_explicit_payloads_falls_back_for_unsupported_endpoint(self):
        server_args = SimpleNamespace(
            warmup_input_lens=[8],
            debug_tensor_dump_input_file=None,
            disaggregation_mode="null",
            dp_size=1,
        )

        payloads = _get_explicit_startup_warmup_payloads(
            server_args, "/v1/chat/completions"
        )

        self.assertIsNone(payloads)


if __name__ == "__main__":
    unittest.main()
