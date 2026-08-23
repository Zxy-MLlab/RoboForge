# Strict code-only zero-shot status — 2026-08-20

## Valid scope

- Active benchmark: LIBERO-Spatial.
- Allowed controller inputs: instruction, RGB-D, camera calibration, and robot
  proprioception.
- Active learned models: none.
- Invalidated learned assets: π0.5 base, π0.5 LIBERO-finetuned, OpenVLA-OFT,
  and the frozen-policy wrapper/selection skill.

## Saved capabilities

- `capability_library/tools/rgbd_pick_place.py`: classical language parsing,
  RGB-D back-projection, table-plane estimation, connected components, Hough
  circle candidates, spatial-relation selection, bounded OSC control, sensor
  projection, and Thea tool registration.
- `capability_library/skills/rgbd-code-pick-place`: validated operating skill.
- Tests: six tool/analysis tests pass; the skill validator passes.

## Development result

LIBERO-Spatial task 0, initial state 0 is development-exposed and non-claimable.
Six generic controller variants all failed after correct source/target selection
and successful Cartesian waypoint tracking. The unresolved stage is grasp
acquisition: the fully closed Panda fingers did not retain the bowl.

The untouched frozen evaluation surface contains 450 episodes across tasks
1-9. The result is 0/450. Of these, 195 failed before action execution because
the classical target detector found no black-bowl candidate; 255 executed the
complete action program but did not satisfy the evaluator. The entire
development-exposed task 0 is excluded.

## Current frontier hypothesis

Classical RGB-D is sufficient for this scene's table localization, circular
object proposal, and the tested spatial relation. The current bottleneck is
contact-aware 6-DoF grasp synthesis under gripper aperture and collision
constraints, not language understanding or Cartesian reachability.
