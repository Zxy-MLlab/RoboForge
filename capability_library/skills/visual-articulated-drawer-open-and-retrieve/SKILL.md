---
name: visual-articulated-drawer-open-and-retrieve
description: Open a language-named drawer using open-vocabulary RGB-D handle and cabinet grounding, visually verify articulation, and re-ground a contained object before retrieval. Use when a target object is in a drawer or when drawer geometry obstructs an otherwise valid grasp.
---

# Visual Articulated Drawer Open and Retrieve

Use only live RGB/RGB-D, camera calibration, language, proprioception, and action history.

1. Trigger this workflow from a drawer named in the live source clause, never from a task ID or benchmark state.
2. Query `drawer handle` and the named cabinet with an open-vocabulary detector. Reject handle boxes larger than a compact local fixture region. If several remain, select the handle nearest the language-grounded contained object. Call `capture_landmark_baseline` with the accepted handle box before motion; retain its opaque `baseline_id`.
3. Estimate the horizontal outward axis as the normalized RGB-D vector from the contained object to the handle; both lie near the drawer translation axis. Fall back to cabinet-center-to-handle only when a contained-object estimate is unavailable, because a large cabinet box is biased under oblique views. Abort articulation when the selected centers are indistinguishable or non-finite.
4. Approach the handle from above, descend with the gripper open, close for a fixed bounded interval, and pull along the observed outward axis. Release and retreat. Use quaternion orientation error for the calibrated top-down pose.
5. Acquire a new RGB-D frame and call `verify_landmark_displacement` with the adapter-owned handle baseline. Require `verified=true`, which means at least 4 cm of observed horizontal displacement, before claiming the drawer opened. Record before/after frames, handle coordinates, cabinet coordinate, commanded direction and observed displacement.
6. Regardless of whether verification passes, never reuse pre-articulation world coordinates. Re-run language grounding, relation selection, SAM and grasp generation on the changed scene. If articulation is not visually verified and the open-vocabulary detector temporarily misses the contained object, use its prior image box only as a prompt for SAM on the new RGB-D frame, then inject the resulting fresh mask/XYZ as a tracked source candidate. Continue retrieval only from fresh sensor estimates.
7. If the handle is absent, the pull is unreachable, or visual displacement is below threshold, emit a structured articulation failure. Do not infer joint state from MuJoCo or use evaluator feedback to decide another action.

Never use reward, done, check_success, BDDL, simulator joint positions, body poses, segmentation IDs, demonstrations, or task/state-specific coordinates.
