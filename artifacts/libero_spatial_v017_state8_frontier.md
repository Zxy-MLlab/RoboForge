# Embodied Intelligence Frontier: LIBERO-Spatial

## Frozen result

Controller `generic_rgbd_closed_loop_pick_place_v017:v001` was evaluated on all 10 tasks at state 8. The batch is sealed: `evaluator_calls=1` per completed episode and `results_consumed_for_iteration=False`.

- Sensor-only closed-loop verification: **4/10**
- Evaluator success: **1/10**
- This is a validated transfer measurement for this controller/state/seed, not a universal intelligence upper bound.

| Task | Sensor | Evaluator | Attachment | Placement | Grasp attempts | Evidence |
|---:|---:|---:|---:|---:|---:|---|
| 0 | fail | fail | pass | fail | 3 | `runs/coding_harness/libero_spatial_v017_state8_full/runs/task0_state8_seed7` |
| 1 | fail | fail | fail | fail | 5 | `runs/coding_harness/libero_spatial_v017_state8_full/runs/task1_state8_seed7` |
| 2 | pass | pass | pass | pass | 1 | `runs/coding_harness/libero_spatial_v017_state8_full/runs/task2_state8_seed7` |
| 3 | pass | fail | pass | pass | 3 | `runs/coding_harness/libero_spatial_v017_state8_full/runs/task3_state8_seed7` |
| 4 | fail | fail | pass | fail | 3 | `runs/coding_harness/libero_spatial_v017_state8_full/runs/task4_state8_seed7` |
| 5 | fail | fail | pass | fail | 2 | `runs/coding_harness/libero_spatial_v017_state8_full/runs/task5_state8_seed7` |
| 6 | pass | fail | pass | pass | 2 | `runs/coding_harness/libero_spatial_v017_state8_full/runs/task6_state8_seed7` |
| 7 | fail | fail | fail | fail | 5 | `runs/coding_harness/libero_spatial_v017_state8_full/runs/task7_state8_seed7` |
| 8 | pass | fail | pass | pass | 1 | `runs/coding_harness/libero_spatial_v017_state8_full/runs/task8_state8_seed7` |
| 9 | fail | fail | fail | fail | 0 | `runs/coding_harness/libero_spatial_v017_state8_full/runs/task9_state8_seed7` |

## Development progression

The Harness authoring loop produced immutable controller versions. Comparable sealed checkpoints are shown below; v009/v010 were failure-replay development batches and are not headline transfer scores.

| Controller | Batch | Sensor | Evaluator | Results fed back? |
|---|---|---:|---:|---:|
| v008 | state 2 full | 6/10 | 5/10 | no |
| v013 | state 3 full | 4/10 | 3/10 | no |
| v016 | state 6 full | 6/10 | 5/10 | no |
| v017 | state 8 full | 4/10 | 1/10 | no |

The v016 result is the best validated frozen score in this workspace so far, while v017 is the latest dependency-consistent candidate. The difference shows state/controller variance; neither establishes a global optimum. Further improvement requires new development states and another sealed transfer batch.

## Frontier interpretation

- Tasks [1, 7, 9] did not obtain sensor-verified attachment. The immediate bottleneck is perception-to-grasp/control compatibility, not evaluator feedback.
- Tasks [0, 4, 5] verified attachment but not placement. Their RGB-D traces show support-transfer/release geometry failures; more grasp ranking alone is insufficient.
- Tasks [3, 6, 8] passed the sensor verifier but failed the evaluator. These are deliberately recorded as verifier/evaluator discrepancies, not silently counted as success.
- Tasks [2] passed the final evaluator in this frozen batch.

## Capability assets

The registry contains the assets for this frozen batch in `capability_library/library.json`, including relation-region projection, strict-plus-calibrated GraspNet retries, post-contact RGB-D/SAM relocalization, language-query fallback, closed-loop recovery, articulated drawer retrieval, success experiences, and frontier-failure records.

Every primary asset declares `current_task_data_used=false` and `privileged_state_used=false`. The controller manifest and runtime dependency hashes are stored under the frozen controller workspace; rollout videos, RGB-D/SAM artifacts, traces, process logs and success-only HDF5 files remain under each run directory.

## Integrity notes

The earlier v013 state-3 task 0 process failure was an external language API disconnect and was not counted as an embodied failure. v014 added a generic lexical noun-query fallback; v015/v016 added bounded fallback grasp candidates and correction retries. No evaluator result was exposed to the authoring agent or used to select an action.
