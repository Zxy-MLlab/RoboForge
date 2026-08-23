# Reporting tracks

Results from these tracks must never be pooled into one number.

## Track A: Harness-acquired task zero-shot (primary)

Frozen learned components are allowed, including VLMs, SAM/DINO, depth and pose
models, learned grasp policies, planners, and robot policies. An asset passes
only when neither its parameters nor its preprocessing/retrieval statistics
were fitted on the evaluated task definitions, demonstrations, or rollouts.

Report two strata. `benchmark-family-disjoint` means no known LIBERO-family
training. `task-disjoint-transfer` permits training on other LIBERO-family
tasks but not the evaluated tasks. Never pool the strata.

Compatibility code may map legal RGB-D, calibration, and proprioception to a
public asset contract. It may not consume evaluator-only data, vary by episode
ID, or manually encode an evaluated task solution.

## Track B: classical diagnostic

Use code-based perception, geometry, inverse kinematics, planning, and feedback
control only. This is an ablation, not the user's intended capability ceiling.
The historical `0/450` LIBERO-Spatial result belongs here.

## Track C: current-task-exposed reference (non-primary)

Any checkpoint, adapter, normalization statistic, or learned retrieval asset
with exposure to an evaluated task belongs here. It may be retained to audit
contamination or reproduce prior systems but cannot answer the primary claim.

## What is being tested

The independent variable is the Harness's access to an open capability market:
internet search, public code, algorithms, pretrained models, documentation, and
reusable experience. The dependent variables are task success and gain over
the initially deployed Harness. Record discovery traces, integration attempts,
registered assets, cross-task reuse, compute/time, and human interventions.
The intended intervention count after an autonomous run starts is zero.

## Selection and reporting

- Use development tasks for failure analysis and autonomous improvement of
  generic tools/skills, but do not fit model parameters on an evaluated task.
- Freeze code, asset hashes, prompts, and runtime parameters before a sealed run.
- Run a sealed surface once per frozen candidate and never iterate from it.
- Keep integration errors separate from valid task failures.
- Report macro and micro success, per-task results, confidence intervals,
  acquisition gain, asset ablations, provenance, compute, and all failures.

## Superseded interpretation

The earlier rule excluding every learned model was a misinterpretation. Its
artifacts remain a valid Track B diagnostic. Suite-wide π0.5/OpenVLA-OFT
checkpoints remain Track C when their training includes evaluated tasks, not
because they are learned models.
