# LIBERO-Spatial strict code-only zero-shot conclusion

## Result

- Frozen claim surface: tasks 1–9, 50 official initial states per task.
- Episodes: 450.
- Success: 0/450 (0.0%).
- Classical perception/tool exceptions: 195/450 (43.3%).
- Full 655-action programs executed without success: 255/450 (56.7%).
- Learned robot or perception models used: none.
- Privileged control inputs used: none.

Task 0 is excluded because it was the development task. Its state 0 was used
to test seven generic controller variants and the Thea integration path; none
succeeded.

## Per-task frozen outcomes

| Task | Success | Perception exception | Executed, manipulation failure |
|---:|---:|---:|---:|
| 1 | 0/50 | 10 | 40 |
| 2 | 0/50 | 3 | 47 |
| 3 | 0/50 | 26 | 24 |
| 4 | 0/50 | 45 | 5 |
| 5 | 0/50 | 19 | 31 |
| 6 | 0/50 | 18 | 32 |
| 7 | 0/50 | 37 | 13 |
| 8 | 0/50 | 2 | 48 |
| 9 | 0/50 | 35 | 15 |

All 450 frozen episodes remain unresolved. Their individual records are under
`runs/frozen_eval/code_rgbd_side_v1_libero_spatial/task_*/state_*/result.json`.

## Capability boundary

The implemented non-learned stack can parse the benchmark's pick/place
language, recover metric geometry from RGB-D, estimate the support plane,
propose circular objects, resolve the tested “between” relation, and track
Cartesian and orientation waypoints. This did not translate into any full
benchmark success.

The first bottleneck is perception robustness: hand-designed Hough and color
criteria fail under location, occlusion, scale, and background changes. The
second and stronger bottleneck is grasp synthesis: even when perception emits
a target and all waypoints execute, fixed center/rim/side grasp families do
not reliably create force closure. Drawer and elevated-support tasks add
fixture interaction, occlusion, collision reasoning, and multi-stage planning.

## Answers to the research questions

1. Under the user's strict “no trained low-level or perception models” rule,
   the measured Thea + code-controller upper bound on the untouched
   LIBERO-Spatial surface is 0/450. This is not a measurement of the strongest
   online LLM: no external LLM credentials were available, and the frozen run
   invoked the registered deterministic tool directly.
2. The agent did construct a reusable capability library: a validated RGB-D
   manipulation skill, a Thea-registered tool, sensor-boundary audit,
   evaluation runners, tests, traces, and failure records. Autonomous asset
   construction is therefore possible, but the acquired assets did not solve
   this benchmark under the imposed model prohibition.
3. The observed frontier is not primarily language parsing. It is robust
   object perception plus contact-aware 6-DoF grasp planning, followed by
   collision-aware fixture interaction and closed-loop recovery. These are
   precisely the capabilities removed when trained perception and robot-policy
   models are prohibited.

## Integrity note

π0.5 base, π0.5 LIBERO-finetuned, and OpenVLA-OFT results are invalidated and
excluded. Frozen outcomes were never used to change the controller. Success,
reward, object-state, BDDL predicates, task IDs, and initial-state IDs were not
available to action selection.
