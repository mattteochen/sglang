# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Profiler span helpers for hot SGLang code paths.

A span has two independent emitters:

* ``record_function`` -- emitted whenever a torch profiler is active, so spans
  show up in torch/Perfetto traces for free (no env, no extra package).
* ``nvtx`` range -- emitted only when the caller opts in via ``nvtx_enabled``
  (wired to a per-subsystem ``SGLANG_ENABLE_NVTX_*`` gate), for Nsight Systems
  timelines. The optional ``nvtx`` package provides colors when installed;
  otherwise the CUDA build of PyTorch provides a dependency-free fallback.

Decoupling the two lets every annotation site -- scheduler stages, batch-overlap
ops, and the speculative-decoding / forward spans -- share one primitive.
"""

import logging
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial, wraps
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_SCHEDULER_NVTX = envs.SGLANG_ENABLE_NVTX_SCHEDULER.get()
_OPERATIONS_NVTX = envs.SGLANG_ENABLE_NVTX_OPERATIONS.get()

# CPU-only PyTorch also exposes the ``torch.cuda.nvtx`` stub, but entering one
# of its ranges raises at runtime. Require a CUDA build before using it as the
# dependency-free fallback.
_TORCH_NVTX_AVAILABLE = (
    getattr(torch.version, "cuda", None) is not None
    and hasattr(torch.cuda, "nvtx")
    and hasattr(torch.cuda.nvtx, "range")
)
_nvtx_module = None
if _SCHEDULER_NVTX or _OPERATIONS_NVTX:
    try:
        import nvtx as _nvtx_module  # type: ignore
    except ImportError:
        if _TORCH_NVTX_AVAILABLE:
            logger.info(
                "An SGLANG_ENABLE_NVTX_* flag is set, but the optional `nvtx` "
                "package is missing. Falling back to torch.cuda.nvtx ranges "
                "without colors."
            )
        else:
            logger.warning(
                "An SGLANG_ENABLE_NVTX_* flag is set, but neither the optional "
                "`nvtx` package nor CUDA-enabled torch.cuda.nvtx is available. "
                "NVTX markers are disabled."
            )

NVTX_AVAILABLE = _nvtx_module is not None or _TORCH_NVTX_AVAILABLE
# Per-subsystem nvtx gates. The record_function path is independent of both.
NVTX_SCHEDULER_ENABLED = _SCHEDULER_NVTX and NVTX_AVAILABLE
NVTX_OPERATIONS_ENABLED = _OPERATIONS_NVTX and NVTX_AVAILABLE

# Default nvtx colors for statically-named spans (only used on the nvtx path).
_NVTX_COLOR_MAP = {
    "scheduler.recv_requests": "blue",
    "scheduler.process_input_requests": "purple",
    "scheduler.get_next_batch_to_run": "green",
    "scheduler.run_batch": "red",
    "scheduler.process_batch_result": "cyan",
}

_NULL_CONTEXT = nullcontext()


def _nvtx_range(debug_name: str, color: Optional[str]):
    if _nvtx_module is not None:
        return _nvtx_module.annotate(debug_name, color=color)
    return torch.cuda.nvtx.range(debug_name)


@contextmanager
def _profile_range_impl(
    debug_name: str, color: Optional[str], record: bool, nvtx_enabled: bool
):
    with ExitStack() as stack:
        if record:
            stack.enter_context(torch.profiler.record_function(debug_name))
        if nvtx_enabled:
            if color is None:
                color = _NVTX_COLOR_MAP.get(debug_name)
            stack.enter_context(_nvtx_range(debug_name, color))
        yield


def profile_range(
    debug_name: str, *, color: Optional[str] = None, nvtx_enabled: bool = False
):
    """Context manager emitting a profiler span for ``debug_name``.

    A torch ``record_function`` is emitted whenever a torch profiler is active;
    an nvtx range is emitted additionally when ``nvtx_enabled`` is true. Returns a
    shared no-op when neither applies, so off-profile hot paths pay only one
    ``_profiler_enabled()`` check.
    """
    record = torch.autograd._profiler_enabled()
    if not record and not nvtx_enabled:
        return _NULL_CONTEXT
    return _profile_range_impl(debug_name, color, record, nvtx_enabled)


def profile_method(
    debug_name: str, *, color: Optional[str] = None, nvtx_enabled: bool = False
):
    """Decorator form of ``profile_range``."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with profile_range(debug_name, color=color, nvtx_enabled=nvtx_enabled):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _scheduler_stage_name(batch) -> str:
    """Return a truthful, stable stage name for a scheduler batch.

    MIXED is kept separate because chunked prefill can contain both extend and
    decode work. Speculative target verification and draft extension are decode
    work if this helper is ever reused below the top-level scheduler boundary.
    """
    mode = batch.forward_mode
    if mode.is_decode() or mode.is_target_verify() or mode.is_draft_extend_v2():
        stage = "decode"
    elif mode.is_mixed():
        stage = "mixed"
    elif mode.is_extend_without_speculative():
        stage = "prefill"
    elif mode.is_idle():
        stage = "idle"
    elif mode.is_prebuilt():
        stage = "prebuilt"
    else:
        stage = getattr(mode, "name", str(mode)).lower()
    return f"sglang.stage.{stage}"


def scheduler_stage_nvtx_method(func):
    """Annotate a whole scheduler forward with its dynamic batch stage."""

    @wraps(func)
    def wrapper(self, batch, *args, **kwargs):
        record = torch.autograd._profiler_enabled()
        if not record and not NVTX_SCHEDULER_ENABLED:
            return func(self, batch, *args, **kwargs)
        with _profile_range_impl(
            _scheduler_stage_name(batch), None, record, NVTX_SCHEDULER_ENABLED
        ):
            return func(self, batch, *args, **kwargs)

    return wrapper


# Pre-bound per-subsystem helpers: torch spans always (under a profiler), nvtx
# ranges only when that subsystem's gate is on.
scheduler_nvtx_method = partial(profile_method, nvtx_enabled=NVTX_SCHEDULER_ENABLED)
operations_nvtx_range = partial(profile_range, nvtx_enabled=NVTX_OPERATIONS_ENABLED)
