# Autonomous Embodied Evolution Smoke

The GPT-5.6-driven evolution loop was run without human intervention on the
LIBERO-Spatial development task 5, state 14, seed 7.

## Evolution trace

- Round 1 created `rgbd_graspnet_closed_loop_v1:v001` and executed it.
- The sensor-only result indicated failure.
- The agent searched public resources and registered the research lead
  `closed_loop_rgbd_placement_correction_2409_09725` with status `discovered`.
- The agent created `rgbd_graspnet_closed_loop_v2:v001` and executed it.
- The second sensor-only result was `sensor_verification_passed`, so the loop
  stopped without opening or using evaluator output.

Trace: `runs/coding_harness/autonomous_task5_state14/round_001/thea_trace.json`
State: `runs/coding_harness/autonomous_task5_state14/evolution_state.json`

## Sealed measurement

The resulting v2 controller was measured separately in a sealed one-episode
batch:

- Sensor verification: `1/1`
- Evaluator success: `0/1`
- Evaluator calls: exactly one
- `results_consumed_for_iteration`: `false`

The evaluator mismatch is retained as a frontier observation. It was not fed
back into the evolution loop.
