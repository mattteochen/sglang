# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/backend.py


import ast
import dataclasses
import logging
import operator
import os
import pprint
import time
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch
import torch.fx as fx
from torch._dispatch.python import enable_python_dispatcher

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compilation_counter import compilation_counter
from sglang.srt.compilation.compiler_interface import EagerAdapter, InductorAdaptor
from sglang.srt.compilation.cuda_piecewise_backend import CUDAPiecewiseBackend
from sglang.srt.compilation.npu_piecewise_backend import NPUPiecewiseBackend
from sglang.srt.compilation.pass_manager import PostGradPassManager
from sglang.srt.environ import envs
from sglang.srt.platforms import current_platform
from sglang.srt.utils.common import is_npu

logger = logging.getLogger(__name__)


def make_compiler(config: CompilationConfig):
    if config.compiler == "eager":
        return EagerAdapter()
    elif config.compiler == "inductor":
        return InductorAdaptor()
    else:
        raise ValueError(f"Unknown compiler: {config.compiler}")


def make_backend(
    graph: fx.GraphModule,
    compile_config: CompilationConfig,
    inductor_config: dict[str, Any],
    graph_pool: Any,
    piecewise_compile_index: int,
    total_piecewise_compiles: int,
    sym_shape_indices: list[int],
    compiled_graph_for_general_shape: Callable,
    sglang_backend,
):

    if current_platform.is_out_of_tree():
        backend_cls = current_platform.get_piecewise_backend_cls()
    elif is_npu():
        backend_cls = NPUPiecewiseBackend
    else:
        backend_cls = CUDAPiecewiseBackend
    return backend_cls(
        graph,
        compile_config,
        inductor_config,
        graph_pool,
        piecewise_compile_index,
        total_piecewise_compiles,
        sym_shape_indices,
        compiled_graph_for_general_shape,
        sglang_backend,
    )


class CompilerManager:
    def __init__(
        self,
        config: CompilationConfig,
    ):
        self.cache = dict()
        self.is_cache_updated = False
        self.compiler = make_compiler(config)

    def compute_hash(self):
        return self.compiler.compute_hash()

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ):
        self.disable_cache = disable_cache
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, "sglang_compile_cache.py")

        if not disable_cache and os.path.exists(self.cache_file_path):
            with open(self.cache_file_path) as f:
                self.cache = ast.literal_eval(f.read())

        self.compiler.initialize_cache(
            cache_dir=cache_dir, disable_cache=disable_cache, prefix=prefix
        )

    def save_to_file(self):
        if self.disable_cache or not self.is_cache_updated:
            return
        printer = pprint.PrettyPrinter(indent=4)
        data = printer.pformat(self.cache)
        with open(self.cache_file_path, "w") as f:
            f.write(data)

    def load(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        runtime_shape: Optional[int] = None,
    ) -> Optional[Callable]:
        handle = self.cache[(runtime_shape, graph_index, self.compiler.name)]
        compiled_graph = self.compiler.load(
            handle, graph, example_inputs, graph_index, runtime_shape
        )
        if runtime_shape is None:
            logger.debug(
                "Directly load the %s-th graph for dynamic shape from %s via "
                "handle %s",
                graph_index,
                self.compiler.name,
                handle,
            )
        else:
            logger.debug(
                "Directly load the %s-th graph for shape %s from %s via " "handle %s",
                graph_index,
                str(runtime_shape),
                self.compiler.name,
                handle,
            )
        return compiled_graph

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs,
        inductor_config: dict[str, Any],
        graph_index: int = 0,
        num_graphs: int = 1,
        runtime_shape: Optional[int] = None,
    ) -> Any:
        if graph_index == 0:
            # before compiling the first graph, record the start time
            global compilation_start_time
            compilation_start_time = time.time()

        compilation_counter.num_backend_compilations += 1

        compiled_graph = None

        # TODO(Yuwei): support cache loading

        # no compiler cached the graph, or the cache is disabled,
        # we need to compile it
        if isinstance(self.compiler, InductorAdaptor):
            maybe_key = None
        else:
            maybe_key = f"artifact_shape_{runtime_shape}_subgraph_{graph_index}"
        compiled_graph, handle = self.compiler.compile(
            graph, example_inputs, inductor_config, runtime_shape, maybe_key
        )

        assert compiled_graph is not None, "Failed to compile the graph"

        # store the artifact in the cache
        if handle is not None:
            self.cache[(runtime_shape, graph_index, self.compiler.name)] = handle
            compilation_counter.num_cache_entries_updated += 1
            self.is_cache_updated = True
            if graph_index == 0:
                # adds some info logging for the first graph
                if runtime_shape is None:
                    logger.info("Cache the graph for dynamic shape for later use")
                else:
                    logger.info(
                        "Cache the graph of shape %s for later use", str(runtime_shape)
                    )
            if runtime_shape is None:
                logger.debug(
                    "Store the %s-th graph for dynamic shape from %s via " "handle %s",
                    graph_index,
                    self.compiler.name,
                    handle,
                )
            else:
                logger.debug(
                    "Store the %s-th graph for shape %s from %s via handle %s",
                    graph_index,
                    str(runtime_shape),
                    self.compiler.name,
                    handle,
                )

        # after compiling the last graph, record the end time
        if graph_index == num_graphs - 1:
            now = time.time()
            elapsed = now - compilation_start_time
            if runtime_shape is None:
                logger.info("Compiling a graph for dynamic shape takes %.2f s", elapsed)
            else:
                logger.info(
                    "Compiling a graph for shape %s takes %.2f s",
                    runtime_shape,
                    elapsed,
                )

        return compiled_graph


@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


# ---------------------------------------------------------------------------
# Whitelist of FX-node targets we can safely fold into an adjacent eager
# (split-op) submodule. The motivation: when only a small amount of compute
# sits between two split ops, that compute often ends up in a 1-or-2-kernel
# CUDA graph whose host overhead exceeds its benefit. Folding those nodes
# into the surrounding eager submodule (a) eliminates the tiny CUDA graph,
# (b) lets the partitioner merge the two split ops into a single
# ``GraphModule.__call__``, and (c) collapses the wrapper-graph
# ``getitem`` tuple-unpacks that would otherwise sit between two eager
# submodule calls.
#
# ``split_graph`` runs on the *Dynamo*-level FX graph (before
# AOTAutograd), so node targets fall into three buckets:
#
#   * ``call_function`` - a Python callable (``operator.getitem``,
#     ``torch.bmm``, ...). Whitelisted in
#     ``_MERGEABLE_CALL_FUNCTION_TARGETS``.
#   * ``call_method`` - a string method name (``"view"``,
#     ``"transpose"``, ``"new_empty"``, ...). Whitelisted in
#     ``_MERGEABLE_METHOD_NAMES``.
#   * ``get_attr`` - model parameter/buffer reads. Always foldable:
#     they only materialise an attribute and launch no work.
#
# When an out-flowing fresh allocation (``new_empty``) gets folded into
# an eager submodule its ``data_ptr()`` would drift across CUDA-graph
# replays. ``_hoist_eager_buffer_allocations`` rewrites every such
# alloc to read from a pre-allocated persistent buffer so the address
# is stable by construction.
# ---------------------------------------------------------------------------
# Kept intentionally minimal: only the targets actually seen sitting
# between two split ops in shipped split-op layouts (today, strictly the
# MLA prefill / decode bridge). Extend these sets if a new bridge pattern
# starts producing tiny standalone compiled submodules between split ops.
_MERGEABLE_CALL_FUNCTION_TARGETS: frozenset = frozenset(
    {
        operator.getitem,
        torch.bmm,
    }
)
_MERGEABLE_METHOD_NAMES: frozenset[str] = frozenset(
    {
        "split",
        "transpose",
        "unsqueeze",
        "view",
        "new_empty",
    }
)


def _is_mergeable_node(node: fx.Node) -> bool:
    """Whether ``node`` can be folded into an adjacent eager submodule."""
    if node.op == "call_function":
        return node.target in _MERGEABLE_CALL_FUNCTION_TARGETS
    if node.op == "call_method":
        return node.target in _MERGEABLE_METHOD_NAMES
    if node.op == "get_attr":
        return True
    return False


def split_graph(
    graph: fx.GraphModule, ops: list[str]
) -> tuple[fx.GraphModule, list[SplitItem]]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    # State machine that walks the FX nodes in original order and decides
    # which ``subgraph_id`` each node belongs to.
    #
    # The base policy is: every split-op call lives in its own eager
    # submodule; every other node lives in the surrounding compiled
    # submodule.
    #
    # The merge policy on top is: whenever we are still "in the wake of"
    # a split op (i.e. since the last split op every intervening node has
    # passed ``_is_mergeable_node``), any subsequent split op joins the
    # same eager submodule as the previous one, and the whitelisted
    # nodes in between are folded into that same eager submodule too.
    #
    # Concretely this collapses patterns like
    #
    #     split_op_A -> [views / bmm / new_empty / ...] -> split_op_B
    #
    # into one eager submodule containing all of them, eliminating both
    # the in-between compiled submodule (and its potential tiny CUDA
    # graph) and the extra ``GraphModule.__call__`` host hop between the
    # two split ops.
    last_split_id: Optional[int] = None
    pending_mergeable: list[fx.Node] = []
    num_adjacent_split_op_merges = 0
    num_mergeable_nodes_absorbed = 0
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        is_split_op = node.op == "call_function" and str(node.target) in ops
        if is_split_op:
            if last_split_id is not None:
                # Strictly adjacent or whitelist-bridged split op. Reuse
                # the previous eager submodule id and fold all bridging
                # nodes into it. Order is preserved by
                # ``keep_original_order=True`` below, so any in-place
                # mutation by the first call is correctly visible to the
                # bridging nodes and the second call (e.g. an indexer
                # mutating ``topk_result`` followed by an attention call
                # that reads it via ``topk_indices``).
                for p in pending_mergeable:
                    node_to_subgraph_id[p] = last_split_id
                num_mergeable_nodes_absorbed += len(pending_mergeable)
                pending_mergeable.clear()
                node_to_subgraph_id[node] = last_split_id
                num_adjacent_split_op_merges += 1
            else:
                subgraph_id += 1
                node_to_subgraph_id[node] = subgraph_id
                split_op_graphs.append(subgraph_id)
                last_split_id = subgraph_id
                subgraph_id += 1
            # Stay merge-eligible: a subsequent split op (with only
            # mergeable nodes between) should still fold.
        else:
            if last_split_id is not None and _is_mergeable_node(node):
                # Provisionally hold this node; we don't know yet whether
                # the next split op is reachable through whitelist-only
                # nodes. If it is, this node will be absorbed into the
                # eager submodule above; if not, we flush it into the
                # next compiled submodule below.
                pending_mergeable.append(node)
            else:
                if last_split_id is not None:
                    # Flush any pending nodes into the next compiled
                    # submodule (they all sit before this non-mergeable
                    # node in FX order, so they belong to the same
                    # subgraph_id as it).
                    for p in pending_mergeable:
                        node_to_subgraph_id[p] = subgraph_id
                    pending_mergeable.clear()
                    last_split_id = None
                node_to_subgraph_id[node] = subgraph_id

    # Tail: if the FX node stream ended on a run of mergeable nodes after
    # a split op, those nodes form a trailing compiled subgraph of their
    # own (there is no following node to share with).
    if pending_mergeable:
        for p in pending_mergeable:
            node_to_subgraph_id[p] = subgraph_id
        pending_mergeable.clear()

    if num_adjacent_split_op_merges or num_mergeable_nodes_absorbed:
        logger.info(
            "split_graph merged %d adjacent split-op call(s) into shared "
            "eager submodules and absorbed %d whitelisted bridging node(s)",
            num_adjacent_split_op_merges,
            num_mergeable_nodes_absorbed,
        )

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    split_gm = torch.fx.passes.split_module.split_module(
        graph, None, lambda node: node_to_subgraph_id[node], keep_original_order=True
    )

    outputs = []

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by intetger graph_id, rather than string name
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


# we share the global graph pool among all the backends
global_graph_pool = None

compilation_start_time = 0.0


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        inductor_config: dict[str, Any],
        graph_pool,
        compile_config: CompilationConfig,
        sglang_backend: "SGLangBackend",
    ):
        super().__init__(module)
        from torch._guards import detect_fake_mode

        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.graph_pool = graph_pool
        self.sglang_backend = sglang_backend
        # When True, it annoyingly dumps the torch.fx.Graph on errors.
        self.extra_traceback = False
        self.inductor_config = inductor_config
        self.compile_config = compile_config

    def run(self, *args):
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)

    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            global compilation_start_time
            compiled_graph_for_dynamic_shape = (
                self.sglang_backend.compiler_manager.compile(
                    submod,
                    args,
                    self.inductor_config,
                    graph_index=index,
                    num_graphs=len(self.compile_submod_names),
                    runtime_shape=None,
                )
            )

            self.module.__dict__[target] = make_backend(
                submod,
                self.compile_config,
                self.inductor_config,
                self.graph_pool,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                compiled_graph_for_dynamic_shape,
                self.sglang_backend,
            )

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


model_tag: str = "backbone"


@contextmanager
def set_model_tag(tag: str):
    """Context manager to set the model tag."""
    global model_tag
    assert (
        tag != model_tag
    ), f"Model tag {tag} is the same as the current tag {model_tag}."
    old_tag = model_tag
    model_tag = tag
    try:
        yield
    finally:
        model_tag = old_tag


# ---------------------------------------------------------------------------
# Eager-submodule allocation hoisting
# ---------------------------------------------------------------------------
# After ``split_graph`` (and the whitelist-based merging it performs), an
# eager submodule may end up containing a fresh-tensor allocation - most
# commonly the attention output buffer:
#
#     output = q.new_empty((s72, 4096))
#     unified_attention_with_output(..., output, ...)   # mutates in-place
#     ... returned to the wrapper graph, consumed by the next compiled
#     submodule as input.
#
# That ``output`` tensor flows into a captured CUDA graph downstream.
# PCG replay reads straight from the ``data_ptr()`` recorded at capture,
# so the eager allocation must land at the same address every forward.
# The CUDA caching allocator is *not* guaranteed to provide that, and in
# practice the first replay does drift versus capture.
#
# Because the set of captured shapes is finite and known at compile time
# (``compile_config.get_capture_sizes()``), we can statically resolve the
# problem: pre-allocate one max-sized buffer per drifting alloc as a
# persistent attribute on the eager submodule, and rewrite the FX graph
# to read ``buf.narrow(sym_axis, 0, runtime_dim)`` instead of allocating
# fresh. ``narrow`` returns a view at the same ``data_ptr()`` as the
# underlying buffer (offset 0), so every replay observes the captured
# address regardless of token count.
# ---------------------------------------------------------------------------

def _node_reaches(start: fx.Node, targets: set[fx.Node]) -> bool:
    """Return True if any of ``targets`` is a transitive user of ``start``."""
    if start in targets:
        return True
    seen: set[fx.Node] = {start}
    stack: list[fx.Node] = list(start.users)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in targets:
            return True
        stack.extend(n.users)
    return False


def _classify_alloc_shape(
    size_arg: Any,
) -> Optional[tuple[int, list[Any]]]:
    """Pick the single dynamic dim of an alloc shape.

    Returns ``(sym_axis, dims_with_sym_node_at_sym_axis)`` where the dim
    at ``sym_axis`` is an ``fx.Node`` and every other dim is a Python
    int. Returns ``None`` if more than one (or zero) dims are dynamic.
    """
    if not isinstance(size_arg, (list, tuple)):
        return None
    sym_axis: Optional[int] = None
    dims: list[Any] = []
    for i, d in enumerate(size_arg):
        if isinstance(d, fx.Node):
            if sym_axis is not None:
                return None  # multiple sym dims not supported
            sym_axis = i
            dims.append(d)
        elif isinstance(d, int):
            dims.append(d)
        else:
            return None
    if sym_axis is None:
        return None
    return sym_axis, dims


def _replace_alloc_with_static_buffer(
    submod: fx.GraphModule, alloc: fx.Node, max_capture_size: int
) -> bool:
    """Rewrite ``alloc`` to read from a pre-allocated persistent buffer.

    Returns ``True`` on success, ``False`` if the node shape is one we
    don't handle yet (so the caller can move on without aborting).
    """
    example = alloc.meta.get("example_value")
    if example is None or not isinstance(example, torch.Tensor):
        return False
    if example.device.type != "cuda":
        return False
    # For ``t.new_empty(size, ...)`` Dynamo emits ``args = (t, size)``.
    if len(alloc.args) < 2:
        return False
    classified = _classify_alloc_shape(alloc.args[1])
    if classified is None:
        return False
    sym_axis, dims = classified
    sym_node = dims[sym_axis]
    assert isinstance(sym_node, fx.Node)

    buf_shape = list(dims)
    buf_shape[sym_axis] = max_capture_size
    buf = torch.empty(tuple(buf_shape), dtype=example.dtype, device=example.device)

    buf_name = f"_pcg_hoisted_buf_{alloc.name}"
    submod.register_buffer(buf_name, buf, persistent=False)

    g = submod.graph
    with g.inserting_before(alloc):
        attr_node = g.get_attr(buf_name)
        attr_node.meta["example_value"] = buf
        view_node = g.call_method(
            "narrow", args=(attr_node, sym_axis, 0, sym_node)
        )
        view_node.meta["example_value"] = example
    alloc.replace_all_uses_with(view_node)
    g.erase_node(alloc)
    return True


def _hoist_outflowing_allocs_in_submod(
    submod: fx.GraphModule, max_capture_size: int
) -> int:
    g = submod.graph
    output_node = next(n for n in g.nodes if n.op == "output")
    output_set: set[fx.Node] = set()
    output_args = output_node.args[0] if output_node.args else ()
    if not isinstance(output_args, (list, tuple)):
        output_args = (output_args,)
    for a in output_args:
        if isinstance(a, fx.Node):
            output_set.add(a)
    if not output_set:
        return 0

    to_hoist: list[fx.Node] = []
    for node in list(g.nodes):
        # Only ``t.new_empty(...)`` is safe to hoist: its contract is "any
        # contents", so reusing the same physical slot across replays
        # preserves semantics (downstream always writes before reads).
        # ``new_zeros`` / ``new_ones`` / ``new_full`` would additionally
        # require a runtime fill we don't synthesize.
        if node.op != "call_method" or node.target != "new_empty":
            continue
        if _node_reaches(node, output_set):
            to_hoist.append(node)

    hoisted = 0
    for alloc in to_hoist:
        if _replace_alloc_with_static_buffer(submod, alloc, max_capture_size):
            hoisted += 1
    if hoisted:
        g.lint()
        submod.recompile()
    return hoisted


def _hoist_eager_buffer_allocations(
    split_gm: fx.GraphModule,
    piecewise_graphs: list[SplitItem],
    max_capture_size: int,
) -> tuple[int, int]:
    """Hoist out-flowing allocs in every eager submodule.

    Returns ``(num_eager_submods_touched, num_allocs_replaced)``.
    """
    if not torch.cuda.is_available():
        return 0, 0
    if max_capture_size <= 0:
        return 0, 0
    touched = 0
    replaced = 0
    for item in piecewise_graphs:
        if not item.is_splitting_graph:
            continue
        submod = getattr(split_gm, item.submod_name)
        n = _hoist_outflowing_allocs_in_submod(submod, max_capture_size)
        if n:
            touched += 1
            replaced += n
    return touched, replaced


class SGLangBackend:

    graph_pool: Any
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: list[int]
    input_buffers: list[torch.Tensor]
    compiler_manager: CompilerManager

    def __init__(
        self,
        config: CompilationConfig,
        graph_pool: Any,
    ):
        assert graph_pool is not None
        self.graph_pool = graph_pool

        self.post_grad_pass_manager = PostGradPassManager()
        self.sym_tensor_indices = []
        self.input_buffers = []

        self.compiler_manager = CompilerManager(config)
        self.inductor_config = {
            "enable_auto_functionalized_v2": False,
        }
        self.compile_config = config

    def configure_post_pass(self):
        self.post_grad_pass_manager.configure()
        self.inductor_config["post_grad_custom_post_pass"] = self.post_grad_pass_manager

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        base_cache_dir = envs.SGLANG_CACHE_DIR.get()

        cache_hash = self.compiler_manager.compute_hash()
        cache_dir = os.path.join(
            base_cache_dir,
            "torch_compile_cache",
            cache_hash,
        )

        os.makedirs(cache_dir, exist_ok=True)
        rank = 0
        dp_rank = 0
        local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}", model_tag)
        os.makedirs(local_cache_dir, exist_ok=True)
        self.compiler_manager.initialize_cache(
            local_cache_dir, disable_cache=False, prefix=""
        )
        compilation_counter.num_graphs_seen += 1

        assert not self._called, "SGLangBackend can only be called once"

        self.graph = graph
        self.configure_post_pass()

        self.split_gm, self.piecewise_graphs = split_graph(
            graph,
            self.compile_config.split_ops,
        )
        from torch._dynamo.utils import lazy_format_graph_code

        # depyf will hook lazy_format_graph_code and dump the graph
        # for debugging, no need to print the graph here
        lazy_format_graph_code("before split", self.graph)
        lazy_format_graph_code("after split", self.split_gm)

        compilation_counter.num_piecewise_graphs_seen += len(self.piecewise_graphs)

        submod_names_to_compile = [
            item.submod_name
            for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        PiecewiseCompileInterpreter(
            self.split_gm,
            submod_names_to_compile,
            self.inductor_config,
            self.graph_pool,
            self.compile_config,
            self,
        ).run(*example_inputs)

        # Stabilise eager-submodule tensors that flow into downstream
        # captured CUDA graphs by replacing in-eager fresh allocations
        # with views of pre-allocated persistent buffers. Done after
        # ``PiecewiseCompileInterpreter.run`` so the rewrite runs on
        # real tensors (not under fake mode), and before the wrapper
        # graph is returned so the captures see the rewritten forwards.
        capture_sizes = self.compile_config.get_capture_sizes()
        if capture_sizes:
            touched, replaced = _hoist_eager_buffer_allocations(
                self.split_gm, self.piecewise_graphs, max(capture_sizes)
            )
            if replaced:
                logger.info(
                    "Hoisted %d out-flowing allocation(s) across %d eager "
                    "submodule(s) to persistent buffers (stable across "
                    "CUDA-graph replays)",
                    replaced,
                    touched,
                )

        rank = torch.distributed.get_rank()

        if rank == 0:
            graph_path = os.path.join(
                local_cache_dir, f"computation_graph_{time.time()}.py"
            )
            if not os.path.exists(graph_path):
                # code adapted from https://github.com/thuml/depyf/blob/dab831108a752d1facc00acdd6d4243891845c37/depyf/explain/patched_lazy_format_graph_code.py#L30 # noqa
                # use `print_readable` because it can include submodules
                src = (
                    "from __future__ import annotations\nimport torch\n"
                    + self.split_gm.print_readable(print_output=False)
                )
                src = src.replace("<lambda>", "GraphModule")
                with open(graph_path, "w") as f:
                    f.write(src)

        self._called = True
        return self.split_gm
