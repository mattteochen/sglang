from typing import Callable, Optional

from torch import nn

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_musa = is_musa()


class MultiPlatformOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method: Callable = self.dispatch_forward()

        # States for torch.compile
        self._original_forward_method = None
        self.is_torch_compile = False
        self._cached_torch_compile_forward_method: Optional[Callable] = None
        self._cached_topk_compile_forward_method: Optional[Callable] = None

    # TODO(mattteochen): cache result
    def _should_keep_moe_cuda_forward_for_torch_compile(self) -> bool:
        try:
            from sglang.srt.layers.moe import get_moe_runner_backend
        except Exception:
            return False
        # DeepSeek grouped prefill compile works by switching MultiPlatformOp
        # modules onto their torch.compile-friendly forwards. For non-routed
        # flashinfer_trtllm MoE, that strategy must keep the CUDA path intact:
        # DeepSeek still calls TopK as a thin wrapper, but the actual routing is
        # consumed by the monolithic flashinfer_trtllm MoE kernel via raw router
        # logits. Switching TopK/FusedMoE to the num_tokens==1 native path would
        # de-fuse that compiled DeepSeek forward strategy.
        return get_moe_runner_backend().is_flashinfer_trtllm()

    def _get_torch_compile_forward_method(self, num_tokens: int) -> Callable:
        if "FusedMoE" in self.__class__.__name__:
            if self._should_keep_moe_cuda_forward_for_torch_compile():
                return self._forward_method
            if num_tokens == 1:
                from sglang.srt.layers.moe.fused_moe_native import (
                    fused_moe_forward_native,
                )

                return fused_moe_forward_native
            return self._forward_method

        if "TopK" in self.__class__.__name__:
            if self._should_keep_moe_cuda_forward_for_torch_compile():
                return self._forward_method
            if num_tokens == 1:
                if self._cached_topk_compile_forward_method is None:
                    self._cached_topk_compile_forward_method = self.forward_native
                return self._cached_topk_compile_forward_method
            return self._forward_method

        if self._cached_torch_compile_forward_method is None:
            self._cached_torch_compile_forward_method = self.forward_native
        return self._cached_torch_compile_forward_method

    def enter_torch_compile(self, num_tokens: int):
        # Skip if Op is already entered compile mode.
        # NOTE(alcanderian): Some Ops(for example RotaryEmbedding) will be reused
        # among layers and `enter_torch_compile` will be called many times.
        # We should prevent `self._original_forward_method` from being overridden when
        # it is not the first time `enter_torch_compile` called.
        if self.is_torch_compile:
            return

        self._original_forward_method = self._forward_method
        self._forward_method = self._get_torch_compile_forward_method(num_tokens)
        self.is_torch_compile = True

    def leave_torch_compile(self):
        # Skip if Op is already exited compile mode.
        if not self.is_torch_compile:
            return

        self._forward_method = self._original_forward_method
        self._original_forward_method = None
        self.is_torch_compile = False

    # Please do not override this method, because `self._forward_method` can change when in torch compile mode
    @debug_kernel_api
    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_npu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_musa(self, *args, **kwargs):
        # XXX (MUSA): MUSA kernels follow the CUDA path by default.
        # At this stage, sgl-kernel support for MUSA is still under active
        # development, so we fall back to the PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        if _is_cuda:
            return self.forward_cuda
        elif _is_hip:
            return self.forward_hip
        elif _is_cpu and _is_cpu_amx_available:
            return self.forward_cpu
        elif _is_npu:
            return self.forward_npu
        elif _is_xpu:
            return self.forward_xpu
        elif _is_musa:
            return self.forward_musa
        else:
            return self.forward_native
