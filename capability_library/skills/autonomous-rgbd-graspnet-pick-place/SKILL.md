---
name: autonomous-rgbd-graspnet-pick-place
description: Solve language-referred tabletop pick-and-place tasks from live RGB-D using Grounding DINO, VLM physical-region selection, SAM, GraspNet, and bounded Cartesian control without simulator object state or manual grasp offsets.
---

# Autonomous RGB-D GraspNet Pick Place

Use only the task language, live RGB-D, camera calibration, and robot
proprioception for action selection. The BDDL path may initialize the benchmark
environment, but its contents and simulator object state must never enter the
planner.

1. Run the frozen Grounding DINO checkpoint on the current RGB frame for nouns
   extracted from the instruction. Save every candidate, not only the maximum.
2. Deproject each box through measured depth and camera calibration. Fuse
   cross-query boxes within 2.5 cm into unique physical regions because one
   object can receive several open-vocabulary labels.
3. Give GPT-5.6-sol the numbered current RGB image, noisy per-region label
   scores, 3D coordinates, workspace center, and instruction. Ask for source,
   destination, and spatial-reference region IDs. Verify `next to`, `between`,
   and `table center` choices with the corresponding 3D relation.
4. Prompt SAM with the selected source box. Feed the resulting target point
   cloud and local RGB-D context to the frozen GraspNet RealSense checkpoint.
5. Reject candidates outside 7 cm of the SAM center or outside the Panda
   gripper width. Prefer approach compatibility >= 0.75; only when none exists,
   fall back to >= 0.55. This tiered rule prevents a high-confidence oblique
   grasp from displacing a stable top-down grasp.
6. Execute approach, descend, close, lift, transfer, place, release, and retreat
   with Cartesian proportional control and a fixed robot top-down tool-axis
   convention. Preserve the visually estimated object-center-to-grasp transform
   during placement using the detector crop's robust RGB-D center, release 3 cm
   above the target grasp pose, retreat, and wait 80 fixed steps for settling.
7. Do not read reward, success, or simulator state while acting. Call the
   evaluator once after execution and write HDF5 only when success is true.

Implementation:

- `../../tools/groundingdino_detector.py`
- `../../tools/sam_box_segment.py`
- `../../tools/graspnet_rgbd_grasp.py`
- `/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py`

Never add a task ID branch, manual instance rank, object pose, simulator mask,
or task-specific TCP offset. Preserve failed runs as evolution evidence.
