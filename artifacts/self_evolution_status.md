# Self-evolution status

## Verified loop

On 2026-08-20, local Qwen2.5-VL-7B-Instruct drove the real Thea Agent loop
through a development-only failure description. The model selected
`search_public_embodied_resources`; Thea executed the Tool, returned public
resource results, and continued to a second model decision. The trace is in
`artifacts/local_qwen_harness_smoke/trace.json` and the summary is in
`artifacts/local_qwen_harness_smoke/report.json`.

The same failure class was then sent through the self-evolution controller.
Query broadening changed an overly specific gripper query into `robot grasp
policy`, discovering:

- `PKU-EPIC/UniDexGrasp`
- `microsoft/UniGraspTransformer`
- `enyen/NewStableTactileGrasp`
- `aadarshram/GRASP`
- `surajitsaikia27/DRL_Continiuos_Control`

Three candidates were explicitly rejected because their current-task
provenance, preprocessing statistics, or Panda embodiment adapter was not
resolved. They remain in the capability library and acquisition ledger.

## Current measured boundary

- The local general VLM can process robot camera images and produce structured
  failure hypotheses/search queries.
- The Thea loop can autonomously search public resources and preserve rejected
  candidates.
- No public grasp candidate has yet passed the provenance plus embodiment gate.
- OpenVLA 7B base now passes the task-disjoint provenance and action-contract
  gate. On LIBERO-Object development task 0/state 0 it loaded successfully,
  generated 7-DoF actions using the public NYU Franka normalization, and the
  simulator accepted 10 actions; the task still failed. See
  `artifacts/dev_openvla_base_libero_object_t0_s0/failure_analysis.json`.
- The LIBERO-Object development runner now saves a valid failure envelope when
  classical RGB-D perception fails; it no longer crashes while trying to capture
  a nonexistent run ID.
- The historical classical LIBERO-Spatial result remains `0/450`; it is not a
  claim about the self-evolving Harness ceiling.

## Frozen next candidate

`artifacts/frozen_openvla_transfer_candidate.yaml` freezes the OpenVLA base
candidate and adapter before sealed evaluation. Its sealed results must not be
used to change the adapter or select another asset.

## Sealed result and defensible boundary

The one-time frozen LIBERO-Object evaluation completed on 500 episodes. The
OpenVLA task-disjoint transfer candidate achieved `0/500` successes, `0/10`
macro task successes, and `0` integration errors. Every episode was a valid
claimable result with `manipulation_execution` failure. The full report is in
`artifacts/sealed_eval_v2_openvla_transfer_report.json` and `.md`.

This is the measured capability boundary of this frozen candidate under the
protocol, with an exact one-sided 95% upper bound of approximately 0.60% on
the episode success rate. It is not a universal upper bound for all future
public assets. Sealed results were not consumed for selection or iteration.

The Harness itself has demonstrated self-evolution: Qwen drove Thea to search
the public internet, provenance-gate resources, and register reusable tools,
skills, and rejected candidates. The resulting capability library is the
auditable answer to whether an agent can construct capability assets.

The dominant remaining bottleneck is closed-loop manipulation: contact-aware
grasping, visual servoing/recovery, and action normalization across robot
embodiments. Checkpoint loading, simulator action contracts, and provenance
integrity were not bottlenecks in the sealed run.

The reusable controller now exposes development-only `integrate_fn` and
`retry_fn` hooks through `make_frontier_registrar`. Each accepted candidate,
integration attempt, and retry result is persisted in the acquisition ledger;
sealed evaluation cannot invoke these hooks.

The complete sealed failure inventory is preserved in
`artifacts/frontier_failures/libero_object_sealed_v2.json` (500/500 valid
episodes, all classified as manipulation execution failures).

CALVIN development evidence now includes a real PyBullet reset/RGB/action
smoke, a `turn_on_led` failure, and a Qwen-driven public-resource search. The
first query produced an irrelevant FTC repository; the search tool was then
updated to prioritize robot visual-servo and manipulation queries. The
follow-up leads remain pending because license and embodiment provenance are
not sufficient for primary-track acceptance. This failed acquisition is
preserved in `artifacts/calvin_failure_driven_acquisition/quality_review.json`.
