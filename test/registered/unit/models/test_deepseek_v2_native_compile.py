import unittest

import torch
from torch import nn

from sglang.srt.models.deepseek_v2 import _validate_native_compile_linear_semantics
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, suite="stage-b-test-1-gpu-small")


class FakeNativeCompileLinear(nn.Module):
    def __init__(
        self,
        *,
        gather_output: bool = False,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.gather_output = gather_output
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.bias = nn.Parameter(torch.zeros(1)) if bias else None


class TestDeepseekV2NativeCompileGuards(unittest.TestCase):
    def test_rejects_gather_output(self):
        module = FakeNativeCompileLinear(gather_output=True)

        with self.assertRaisesRegex(RuntimeError, "gather_output=True"):
            _validate_native_compile_linear_semantics(
                module, module_name="q_proj", allow_bias=False
            )

    def test_rejects_input_split_inside_row_parallel_wrapper(self):
        module = FakeNativeCompileLinear(input_is_parallel=False)

        with self.assertRaisesRegex(RuntimeError, "input_is_parallel=False"):
            _validate_native_compile_linear_semantics(
                module, module_name="o_proj", allow_bias=True
            )

    def test_rejects_skip_bias_add(self):
        module = FakeNativeCompileLinear(skip_bias_add=True)

        with self.assertRaisesRegex(RuntimeError, "skip_bias_add=True"):
            _validate_native_compile_linear_semantics(
                module, module_name="q_proj", allow_bias=False
            )

    def test_allows_bias_only_when_explicitly_supported(self):
        module = FakeNativeCompileLinear(bias=True)

        with self.assertRaisesRegex(RuntimeError, "with bias"):
            _validate_native_compile_linear_semantics(
                module, module_name="down_proj", allow_bias=False
            )

        _validate_native_compile_linear_semantics(
            module, module_name="o_proj", allow_bias=True
        )


if __name__ == "__main__":
    unittest.main()
