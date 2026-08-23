---
name: select-frozen-embodied-policy
description: Select, classify, and freeze learned perception, VLM, grasp, planning, or robot-policy assets without benchmark leakage. Use when discovering a pretrained embodied asset, deciding whether its training provenance permits task-zero-shot use, or preparing a candidate for LIBERO or another sealed benchmark evaluation.
---

# Select Frozen Embodied Policy

## Procedure

1. Record the exact repository, revision, checkpoint hash, license, embodiment,
   observation keys, action space, all known training datasets, preprocessing
   statistics, adapters, and claimed benchmarks.
2. Classify the asset using `references/tracks.md`. If exposure is unknown,
   choose the more contaminated track until evidence resolves it.
3. Allow frozen learned models when their provenance excludes the currently
   evaluated tasks. Reject them from the primary track when provenance is
   unknown or includes those tasks' demonstrations, images, rollouts, labels,
   adapters, or action-normalization statistics. Disclose training on other
   tasks in the same benchmark family as task-disjoint transfer.
4. Reject assets requiring privileged state, goal predicates, reward, success,
   or evaluation-state-specific configuration at inference time.
5. Make all compatibility choices on a non-claimable development manifest:
   image orientation and size, state/action adapters, action chunk length, and
   deterministic inference seed.
6. Freeze the asset and adapter configuration before reading sealed
   evaluation outcomes. Hash both and append them to the capability registry.
7. Run a sealed manifest only once for a frozen candidate. Never choose
   checkpoints, prompts, tools, or hyperparameters from sealed performance.
8. Report tracks separately. Do not label a LIBERO-trained component as task
   zero-shot merely because no training occurred during the current run.

## Failure handling

- Treat import, shape, normalization, renderer, and action-bound errors as
  integration failures, not task failures.
- A valid rollout that reaches the horizon without the evaluator-only success
  predicate is a task failure.
- Preserve all valid failed episodes for taxonomy analysis, but never turn
  development or evaluation rollouts into learned task-specific assets.

Read `references/tracks.md` before classifying a new asset.
