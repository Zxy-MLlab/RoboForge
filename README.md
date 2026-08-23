# RoboForge: Embodied Intelligence Frontier Harness

This workspace evaluates whether a Thea-style embodied harness can expand its
capabilities by discovering and registering public models, algorithms, tools,
and skills. The first benchmark is LIBERO.

The canonical autonomous entry point is
`evaluation/run_embodied_codex_libero.py`. One command runs sensor-only
development, public capability acquisition, immutable Task Skill freezing,
deterministically selected unseen-state validation, and post-batch sealed
scoring. GPT-5.6 authors the complete `run(robot)` controller and owns its
loops, branches, reobservation, recovery, and Tool composition. The environment
adapter owns only sensors, bounded actions, and the evaluator isolation
boundary.

For example:

```bash
python evaluation/run_embodied_codex_libero.py \
  --tasks 4 --development-state 23 \
  --output runs/embodied_codex/libero_spatial_task4
```

`evaluation/run_autonomous_evolution.py` and
`evaluation/run_task_skill_validation.py` are internal development and frozen
validation stages used by that entry point. The old controller-spec path is
not part of the canonical Embodied Codex workflow.

## Integrity boundary

- The agent may observe the natural-language instruction, benchmark-approved
  RGB-D cameras, documented camera calibration, and robot proprioception.
- LIBERO success predicates, object poses, simulator internals, BDDL goal
  predicates, rewards, and evaluation labels are evaluator-only data.
- Evaluation episodes are never converted into demonstrations or training
  data. Failed test-state IDs may be reported, but may not select or tune
  episode-specific actions, prompts, checkpoints, or hyperparameters.
- Learned robot-policy and learned-perception checkpoints are allowed in the
  primary track when their training and preprocessing provenance is documented
  and disjoint from the evaluated task. Checkpoints without such evidence
  remain rejected audit assets.
- Every claimed improvement must be rerun on a frozen evaluation manifest and
  include per-episode traces. Development uses a disjoint smoke manifest.

See `protocol/anti_cheating.yaml` for the machine-readable policy.

## Current LIBERO-Spatial frontier

The complete-program autonomous Harness has final success evidence for task
types 3, 5, and 7. Task 4 (open the top drawer, retrieve the bowl, and place it
on the plate) remains under autonomous development. There is not yet a valid
10-task score or a three-unseen-state `sensor_validated` Task Skill, so the
project does not yet claim a LIBERO-Spatial capability upper bound.

Older v016--v019 controllers and their 6/10-style measurements were produced
under a materially different, externally engineered workflow. They remain
historical baselines and must not be reported as the autonomous Embodied Codex
result. All new claims must follow `protocol/reporting_tracks.md` and include
the campaign seal, sensor traces, frozen Skill hashes, and sealed evaluator
files.

## Cross-benchmark status

The first benchmark remains LIBERO and its new autonomous campaign is still in
progress. A
benchmark-neutral CALVIN adapter is protocol-tested in
`Thea/simulation/thea_simulation/adapters/calvin.py`; it accepts official
CALVIN environment objects, projects RGB observations, and keeps evaluator
success outside the agent observation. The server currently lacks CALVIN's
official evaluation trajectory/task-sequence files, so no CALVIN score is
claimed. See `manifests/calvin_zero_shot_smoke.json` for the explicit data gap.
