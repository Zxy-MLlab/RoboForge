---
name: rgbd-code-pick-place
description: Execute generic LIBERO-style pick-and-place tasks with classical RGB-D geometry, connected components, circle voting, bounded Cartesian feedback, and Panda gripper control. Use for strict zero-shot robotics where learned policy and learned perception checkpoints are forbidden.
---

# RGB-D Code Pick Place

1. Parse only the natural-language instruction with `parse_pick_place_instruction`.
2. Project the allowed upright RGB-D camera image through the documented camera
   intrinsics and extrinsics. Estimate the support plane from the robust depth
   histogram; never read object-state or BDDL fields.
3. Detect candidate rims with classical Hough voting and resolve source/target
   through appearance and generic spatial relations. Keep all thresholds
   workspace-relative and independent of task or initial-state identifiers.
4. Execute bounded OSC_POSE Cartesian waypoints using only end-effector
   proprioception. Open before approach, close for the full Panda gripper
   travel, lift, transport, release, and retreat.
5. Discard reward, done, success, termination, and evaluator state while
   selecting actions. Read evaluator success only after the trajectory ends.
6. Save the instruction, sensor provenance, selected geometric candidates,
   action trace, video, and a failure taxonomy entry for every run.

Use `../../tools/rgbd_pick_place.py` as the implementation and run its tests
before changing thresholds or control phases. Treat a failed episode as
failure-taxonomy evidence, never as a training demonstration or a reason to
write a task-ID or initial-state-ID special case.
