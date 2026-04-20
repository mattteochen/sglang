# DeepSeek Forward Investigation Plan

Date: 2026-04-20

## Goal

Understand why DeepSeek/GLM experimental prefill compile selects
`nested_compile_region` for later layers, but does not emit
`__hierarchical_compile` logs or show clear `invoke_subgraph` evidence in the
full SGLang server path.

## Current Findings

1. `TORCH_LOGS='+hierarchical_compile,recompiles'` is working.
   `test.py` emits `__hierarchical_compile` logs as expected.

2. DeepSeek runtime target selection is working.
   The latest `log.txt` shows:
   - layers `0-2`: `raw`
   - layers `3-7`: `wrapped(...)`

3. The server path still shows:
   - many `__recompiles`
   - zero `__hierarchical_compile`

4. `test2.py` proves that when hierarchical compile is entered but reuse
   fails, PyTorch still prints `__hierarchical_compile` diagnostics.
   That means the SGLang absence is meaningful.

5. `test3.py` reproduces the same shape of problem in a smaller setting:
   - top-level model is `torch.compile`d
   - model loops over layers
   - per-layer target is resolved through
     `_get_nested_compile_region_target(...)`
   - both `ModuleList` and named-attribute storage behave the same
   - no `__hierarchical_compile` appears

6. The current best hypothesis is:
   the whole-model outer loop plus dynamic bound-method target resolution is
   preventing Dynamo from preserving the nested region as hierarchical compile /
   `invoke_subgraph`.

## Files Touched

- `python/sglang/srt/models/deepseek_v2.py`
  - runtime target-resolution logging
  - temporary `invoke_subgraph` probe
- `test2.py`
  - heterogeneous layers, reuse should fail but hierarchical logs should still appear
- `test3.py`
  - whole-model loop repro with raw vs wrapped target resolution

## Immediate Action Item On DeepSeek Forward

Refactor or locally prototype the forward path so the compiled outer model calls
the layer directly, instead of routing through
`_get_experimental_prefill_nested_compile_region_target(layer)(...)` inside the
loop.

The key question:

Would `hierarchical_compile` appear if the later layers expose the nested region
through a direct call shape such as `x = layer(x, ...)`, where the wrapped
behavior is attached to the layer method itself rather than selected dynamically
inside the compiled loop?

## Tomorrow's Plan

1. Build `test4.py`.
   Requirements:
   - keep the whole-model compiled outer loop
   - keep the same "layers 0-2 raw, 3+ wrapped" behavior
   - do **not** use `_get_nested_compile_region_target(...)` in the loop
   - instead, make the wrapped path reachable through a direct method call

2. Run `test4.py` with:

   ```bash
   TORCH_LOGS='+hierarchical_compile,recompiles' python test4.py
   ```

3. Interpret the result.
   - If `__hierarchical_compile` appears:
     the likely blocker is the dynamic target-resolution pattern in the
     DeepSeek forward loop.
   - If it still does not appear:
     the problem is likely deeper in how the full outer graph is being traced or
     in how `nested_compile_region` interacts with this style of module call.

4. Rerun the real server with the current probe still enabled and search for:

   ```bash
   rg -n "invoke_subgraph probe|runtime target resolution|__hierarchical_compile" log.txt
   ```

5. Check whether the probe fires.
   - If runtime target resolution appears but `invoke_subgraph probe entered`
     does not:
     Dynamo is likely never entering
     `InvokeSubgraphHigherOrderVariable._call_function` for this path.

6. If `test4.py` supports the direct-call hypothesis, patch DeepSeek locally.
   Possible direction:
   - move the nested-compile-region wrapping closer to the layer method
   - avoid per-iteration dynamic method rebinding in the compiled loop

7. Re-run the server and compare:
   - compile latency
   - presence of `__hierarchical_compile`
   - whether compile is flatter / more reusable

## Useful Commands

### Existing small repros

```bash
TORCH_LOGS='+hierarchical_compile,recompiles' python test.py --num-layers 10
TORCH_LOGS='+hierarchical_compile,recompiles' python test2.py --num-layers 10
TORCH_LOGS='+hierarchical_compile,recompiles' python test3.py --storage both --num-layers 8 2>&1 | tee test3.log
```

### Current server grep

```bash
rg -n "nested_compile_region|invoke_subgraph probe|__hierarchical_compile|__recompiles" log.txt
```

## Notes

- `ModuleList` itself does not currently look like the differentiator.
- The likely pressure point is the compiled whole-model loop shape.
- Keep the temporary probe until `test4.py` and one more server run settle
  whether `invoke_subgraph` is ever reached.
