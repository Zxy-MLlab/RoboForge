# Embodied Harness LIBERO-Spatial Status

## What was built

Thea now operates as a Coding-Agent-style embodied harness. It loads reusable
Skills, exposes public Tools, asks an LLM to create an immutable standalone
Controller, executes that Controller in LIBERO, returns sensor-only evidence,
and keeps evaluator results behind a scoring barrier. A new Controller version
is required whenever a runtime dependency changes.

The primary controller uses general public capabilities: GPT/VLM reasoning,
GroundingDINO, SAM, RGB-D geometry, GraspNet, guarded motion, visual grasp
verification, visual placement verification, bounded correction, and a visual
drawer-retrieval Skill. No model trained on the evaluated task, BDDL goal,
reward, `done`, `check_success`, simulator pose, or body identity is supplied
to the controller.

## Frozen measurements

| Frozen controller | Unseen state | Tasks | Sensor-only verification | LIBERO evaluator |
|---|---:|---:|---:|---:|
| `generic_rgbd_closed_loop_pick_place_v016:v001` | 6 | 10 | 6/10 | 5/10 |
| `generic_rgbd_closed_loop_pick_place_v017:v001` | 8 | 10 | 4/10 | 1/10 |
| `generic_rgbd_closed_loop_pick_place_v018:v001` | 10 | 10 | 3/10 | 6/10 |
| `generic_rgbd_closed_loop_pick_place_v019:v001` | 12 | 10 | 5/10 | 5/10 |

Evidence:

- [v016 sealed summary](../runs/coding_harness/libero_spatial_v016_state6_full/summary.json)
- [v017 sealed summary](../runs/coding_harness/libero_spatial_v017_state8_full/summary.json)
- [v016 frontier report](libero_spatial_v016_state6_frontier.md)
- [v017 frontier report](libero_spatial_v017_state8_frontier.md)
- [v018 sealed summary](../runs/coding_harness/libero_spatial_v018_state10_full/summary.json)
- [v019 sealed summary](../runs/coding_harness/libero_spatial_v019_state12_full/summary.json)
- [v018 frontier report](libero_spatial_v018_state10_frontier.md)
- [v019 frontier report](libero_spatial_v019_state12_frontier.md)

Every episode in all four batches has exactly one evaluator call, and all
manifests state `results_consumed_for_iteration=false`. v018 is the best
validated frozen result observed in this workspace at 6/10 evaluator success;
v019 is the latest dependency-consistent candidate at 5/10 on a new state-12
batch. The difference is evidence of state and controller variance, not a
universal intelligence upper bound.

## Capability library

`capability_library/library.json` contains 81 provenance-checked assets,
including relation-region projection, strict-plus-calibrated GraspNet retry,
source relocalization, language-query fallback, closed-loop recovery, drawer
retrieval, successful trajectories, and frontier-failure records. All primary
assets declare `current_task_data_used=false` and `privileged_state_used=false`.

## Current frontier

The recurring failures are:

- attachment failure after bounded grasp candidates, indicating perception to
  grasp/control compatibility limits;
- attachment succeeds but placement fails, indicating transfer/release/support
  geometry limits;
- sensor verifier passes while evaluator fails, indicating unresolved visual
  instance identity or relation-verification errors;
- large variation between unseen initial states, indicating weak cross-state
  generalization.

The current experiment therefore measures a real, auditable capability boundary,
not a claim of general-purpose robot intelligence. The next useful benchmark
should target the unresolved relation/instance, placement geometry, and
cross-state tasks listed in the per-episode frontier records.

## Verification

The focused Harness/tool suite passes 31 tests. The v017 controller source and
runtime dependency hashes are sealed in
`runs/coding_harness/controllers_frozen_v15/generic_rgbd_closed_loop_pick_place_v017/v001`.
