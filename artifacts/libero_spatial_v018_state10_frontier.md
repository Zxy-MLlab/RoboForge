# Embodied Intelligence Frontier: LIBERO-Spatial

## Frozen result

Controller `generic_rgbd_closed_loop_pick_place_v018:v001` was evaluated on all 10 tasks at state 10. The batch is sealed: `evaluator_calls=1` per completed episode and `results_consumed_for_iteration=False`.

- Sensor-only closed-loop verification: **3/10**
- Evaluator success: **6/10**
- This is a validated transfer measurement for this controller/state/seed, not a universal intelligence upper bound.

| Task | Sensor | Evaluator | Attachment | Placement | Grasp attempts | Evidence |
|---:|---:|---:|---:|---:|---:|---|
| 0 | pass | pass | pass | pass | 2 | `runs/coding_harness/libero_spatial_v018_state10_full/runs/task0_state10_seed7` |
| 1 | fail | pass | pass | fail | 1 | `runs/coding_harness/libero_spatial_v018_state10_full/runs/task1_state10_seed7` |
| 2 | pass | pass | pass | pass | 1 | `runs/coding_harness/libero_spatial_v018_state10_full/runs/task2_state10_seed7` |
| 3 | fail | fail | pass | fail | 4 | `runs/coding_harness/libero_spatial_v018_state10_full/runs/task3_state10_seed7` |
| 4 | pass | pass | pass | pass | 1 | `runs/coding_harness/libero_spatial_v018_state10_full/runs/task4_state10_seed7` |
| 5 | fail | fail | pass | fail | 4 | `runs/coding_harness/libero_spatial_v018_state10_full/runs/task5_state10_seed7` |
| 6 | fail | pass | pass | fail | 1 | `runs/coding_harness/libero_spatial_v018_state10_full/runs/task6_state10_seed7` |
| 7 | fail | fail | pass | fail | 1 | `runs/coding_harness/libero_spatial_v018_state10_full/runs/task7_state10_seed7` |
| 8 | fail | fail | pass | fail | 4 | `runs/coding_harness/libero_spatial_v018_state10_full/runs/task8_state10_seed7` |
| 9 | fail | pass | pass | fail | 1 | `runs/coding_harness/libero_spatial_v018_state10_full/runs/task9_state10_seed7` |

## Development progression

The Harness authoring loop produced immutable controller versions. Comparable sealed checkpoints are shown below; v009/v010 were failure-replay development batches and are not headline transfer scores.

| Controller | Batch | Sensor | Evaluator | Results fed back? |
|---|---|---:|---:|---:|
| v008 | state 2 full | 6/10 | 5/10 | no |
| v013 | state 3 full | 4/10 | 3/10 | no |
| v016 | state 6 full | 6/10 | 5/10 | no |
| v017 | state 8 full | 4/10 | 1/10 | no |
| v018 | state 10 full | 3/10 | 6/10 | no |

The v018 result is the best validated frozen score in this workspace so far. The spread across v016-v018 still shows substantial state/controller variance; no checkpoint establishes a global optimum. Further improvement requires new development states and another sealed transfer batch.

## Frontier interpretation

- Tasks none did not obtain sensor-verified attachment. The immediate bottleneck is perception-to-grasp/control compatibility, not evaluator feedback.
- Tasks [1, 3, 5, 6, 7, 8, 9] verified attachment but not placement. Their RGB-D traces show support-transfer/release geometry failures; more grasp ranking alone is insufficient.
- Tasks none passed the sensor verifier but failed the evaluator. These are deliberately recorded as verifier/evaluator discrepancies, not silently counted as success.
- Tasks [0, 1, 2, 4, 6, 9] passed the final evaluator in this frozen batch.

## Capability assets

The registry contains the assets for this frozen batch in `capability_library/library.json`, including relation-region projection, strict-plus-calibrated GraspNet retries, post-contact RGB-D/SAM relocalization, language-query fallback, closed-loop recovery, articulated drawer retrieval, success experiences, and frontier-failure records.

Every primary asset declares `current_task_data_used=false` and `privileged_state_used=false`. The controller manifest and runtime dependency hashes are stored under the frozen controller workspace; rollout videos, RGB-D/SAM artifacts, traces, process logs and success-only HDF5 files remain under each run directory.

## Integrity notes

The earlier v013 state-3 task 0 process failure was an external language API disconnect and was not counted as an embodied failure. v014 added a generic lexical noun-query fallback; v015/v016 added bounded fallback grasp candidates and correction retries. No evaluator result was exposed to the authoring agent or used to select an action.
