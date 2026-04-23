# DeepSeek Prefill Grouped-Compile Recompilation Investigation

## Scope

This note documents the `torch.compile` recompiles observed in `server_log3.txt`
for grouped DeepSeek prefill compilation after the server printed:

```text
[2026-04-23 03:54:17] The server is fired up and ready to roll!
```

The relevant compiled function is `_forward_prefill_group_impl(...)` in
`python/sglang/srt/models/deepseek_v2.py`.

## Executive Summary

The post-`ready` recompiles are real. They are not caused by a single missing
warmup length. They come from a combination of:

- New batched NSA metadata layouts at runtime, especially `page_table_1.shape[0]`
  changing from the startup-warmed `1` to runtime values like `3` and `2`
- Structural graph specialization on layer-group topology
  (`DeepseekV2MLP` vs `DeepseekV2MoE`)
- Compile-time branching on communication decisions such as
  `should_fuse_mlp_allreduce_with_next_layer()`
- Attribute-based guards on `hidden_states._sglang_needs_allreduce_fusion`
- A few remaining shape and threshold guards such as `residual.shape[0]`,
  the `<= 2048` FlashInfer fusion threshold, and the custom-AR size threshold

`page_table_1` dim0 is already marked dynamic in the grouped prefill path, so
making that one tensor dynamic is not sufficient to eliminate these recompiles.

## What Happened After `ready`

The startup command in `server_log3.txt` used:

```text
warmup_batch_input_lens=[[1], [8], [16], [32], [64], [256], [1024], [1024, 1024, 1024, 1024]]
```

That warmup covers:

- `bs=1` fused-family prefills
- `bs=4` unfused-family prefills

It does not cover batched fused-family prefills such as `bs=2` or `bs=3` with
total tokens `<= 2048`.

After `ready`, rank 0 produced these new `_forward_prefill_group_impl`
variants:

- `[3/15]` at `server_log3.txt:5223`
- `[3/16]` at `server_log3.txt:6288`
- `[3/17]` at `server_log3.txt:6656`
- `[3/18]` at `server_log3.txt:7862`
- `[3/19]` at `server_log3.txt:8910`

The first clearly new runtime family was a batched fused prefill. The guard
failures show:

- `page_table_1` dim0 changed from `1` to `3`
- flattened token count landed below the `2049` unfused cutoff
- token shapes changed from singleton warmup shapes to real batched runtime
  shapes

Later the benchmark also introduced another batched layout where
`page_table_1.shape[0] == 2`.

## Distinct Guard Families Seen Post-`ready`

The raw log repeats many guards, but they collapse into a small number of root
causes.

### 1. Batched NSA metadata layout guards

Examples from `server_log3.txt`:

- `page_table_1 ... expected 1, actual 3`
- `page_table_1 ... expected 1, actual 2`

Interpretation:

- Startup warmed `bs=1` and `bs=4`
- Runtime hit new batched fused layouts with `bs=3` and later `bs=2`
- In NSA prefill, `page_table_1` is batched metadata, so its dim0 changes with
  the number of merged requests in the prefill batch

Important nuance:

- `page_table_1` dim0 is already passed through
  `torch._dynamo.maybe_mark_dynamic(...)`
- This means the observed recompiles are not explained by `page_table_1` alone
- Other guards in the same trace variant still force recompilation

## 2. Layer-group topology guards

Examples from `server_log3.txt`:

- `DeepseekV2MoE`
- `DeepseekV2MLP`

Interpretation:

- A single generic compiled body is reused across heterogeneous layer groups
- Dynamo guards on the concrete module types reachable through
  `self.layers[i].mlp`
- When runtime execution reaches a different group topology, a new graph is
  compiled

This is one of the largest remaining recompile sources.

## 3. Communication branch guards

Examples from `server_log3.txt`:

- `self.layers[1].layer_communicator.is_last_layer == True`
- `self.layers[1].layer_communicator.is_last_layer == False`
- `2049 <= forward_batch.input_ids.size()[0]`
- `12288*hidden_states.size()[0] > 8388608`

Interpretation:

- `_forward_prefill_group_impl(...)` currently calls
  `should_fuse_mlp_allreduce_with_next_layer(...)` inside the compiled loop
- That helper branches on runtime token count and `is_last_layer`
- The allreduce path also exposes a size threshold used to decide whether
  custom allreduce is eligible

This means communication-policy decisions are currently part of the traced
graph shape.

## 4. Attribute-presence guards

Example from `server_log3.txt`:

- `not hasattr(hidden_states, '_sglang_needs_allreduce_fusion')`

Interpretation:

- The compiled path specializes on whether `hidden_states` carries the
  `_sglang_needs_allreduce_fusion` marker
- This shows up in `prepare_attn(...)` and creates a new graph family when the
  attribute presence differs

## 5. Remaining tensor-shape guards

Examples from `server_log3.txt`:

- `positions ... expected 1, actual 1218`
- `residual ... expected 1, actual 1218`
- `residual ... expected 1, actual 527`

Interpretation:

- `positions` and `hidden_states` dim0 are already marked dynamic
- `residual` is not currently marked dynamic in the grouped prefill input
  marking helper
- Some of these shape guards also appear together with the higher-impact
  structural guards above

## Why `ready` Did Not Prevent This

`ready` only means the configured startup warmups completed. It does not mean
that Dynamo has seen every runtime family the scheduler can produce.

In this run, warmup completed after covering:

- singleton fused prefills
- one batched unfused prefill (`4x1024`)

The benchmark then introduced new batched fused families that warmup did not
exercise.

## What Can Be Solved With Warmup Alone

Warmup can reduce first-hit recompilation for common runtime families, but it
cannot remove structural graph specialization by itself.

Practical warmup extension:

- Keep the existing unfused batched warmup such as `4x1024`
- Add one or more fused batched warmups, for example:
  - `3x512`
  - `2x512`

Why:

- `3x512` warms a fused batched family with `bs=3`, total tokens `1536`
- `2x512` warms a fused batched family with `bs=2`, total tokens `1024`

This directly targets the runtime batch layouts that appeared after `ready`.

## What Requires Code Changes

### Priority 1: Hoist communication decisions out of the compiled loop

Current issue:

- `_forward_prefill_group_impl(...)` computes:
  - `should_fuse_mlp_allreduce_with_next_layer(...)`
  - `should_use_reduce_scatter(...)`
- This causes Dynamo to guard on token-count thresholds and `is_last_layer`

Recommended change:

- Precompute these booleans before entering the compiled body
- Pass them in as inputs, or construct group-specific constants outside the
  traced region

Expected effect:

- Removes a major source of recompilation tied to communication policy

### Priority 2: Normalize `_sglang_needs_allreduce_fusion`

Current issue:

- The graph specializes on the presence or absence of the Python attribute
  `_sglang_needs_allreduce_fusion`

Recommended change:

- Replace attribute-presence checks with an explicit boolean input or a tensor
  flag that is always present

Expected effect:

- Eliminates one full guard family

### Priority 3: Stop sharing one generic compiled body across heterogeneous group topologies

Current issue:

- Different groups contain different mixes of `DeepseekV2MLP` and
  `DeepseekV2MoE`
- Dynamo recompiles when the group topology changes

Recommended change:

- Compile per fixed group topology, or instantiate separate compiled runners
  for different topology signatures

Expected effect:

- Removes repeated topology-driven recompiles

### Priority 4: Mark `residual` dynamic

Current issue:

- `residual` still shows singleton-to-runtime size guards

Recommended change:

- Extend `_mark_experimental_prefill_dynamic_inputs(...)` to also mark
  `residual` dim0 dynamic

Expected effect:

- Reduces one remaining shape-only guard family

This is useful, but it is not the main lever compared with the structural items
above.

## Recommended Order Of Work

1. Extend warmup coverage to include representative fused batched cases
2. Mark `residual` dynamic
3. Hoist communication-policy decisions out of the compiled loop
4. Replace `_sglang_needs_allreduce_fusion` attribute guards with explicit
   stable inputs
5. Split compiled runners by group topology if recompiles still matter

## Bottom Line

The recompiles observed after `ready` are expected given the current design.
They happened because runtime introduced batch families and control-flow
combinations that startup warmup did not cover, and because the grouped prefill
compiled function still specializes on several Python-level structural and
communication decisions.

Warmup can reduce first-hit recompilation. It cannot fully solve the problem.
To materially reduce recompilation frequency, the compile boundary needs to be
made less sensitive to:

- batch metadata layout
- layer-group topology
- communication-policy branching
- Python attribute presence
