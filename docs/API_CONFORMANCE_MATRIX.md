# ASPIRE / CaP-X LIBERO Controller API conformance

RoboForge exposes these methods as ordinary Controller Python methods through
the single `sdk` RPC operation. They are not OpenHands LLM tools. Source
revisions are ASPIRE `f4c8939aab0af9b97690c561bd80e282940f7886` (Apache-2.0/MIT)
and CaP-X `53e9966d7a8e2fa7494676772bccc35280f5c0ed` (MIT).

The authoritative per-function records are in
[`API_CONFORMANCE_MATRIX.json`](API_CONFORMANCE_MATRIX.json). Each row names
the exact upstream file, local implementation, signature, dependencies, test,
live evidence, semantic comparison and status.

Overall status is **partial conformance**: all five upstream stacks are
installed, deployed and live-invoked, but Molmo point grounding has not yet
produced an upstream-parseable coordinate on the real LIBERO frame. Composite
language-to-pose and language-to-grasp rows therefore remain partial.

Current evidence summary:

- Real LIBERO task 0/state 0 observation, camera-frame math, depth/point
  transforms, OBB, quaternion helpers, blocking home/joint/gripper control:
  verified in `/root/autodl-tmp/experiments/api-conformance-task0-state0-20260903/api-conformance.json`.
- Contact-GraspNet, PyRoKi, cuRobo and SAM3 have live service evidence in
  `/root/autodl-tmp/experiments/api-live-evidence-20260903/final` and
  `/root/autodl-tmp/experiments/sam3-live-response.json` (real LIBERO RGB-D,
  pose and RGB segmentation requests with HTTP 200 responses and decoded
  native outputs). Molmo2 is also deployed from the Apache-2.0
  `tollea1234/Molmo2-4B-FP8` Transformers checkpoint and executes real CUDA
  inference for LIBERO RGB frames. Its current output is not reliably emitted
  in the upstream point-coordinate format, so point grounding rows are marked
  `partially_verified` in the JSON matrix; no mock result is used.
- Dual-arm rows report explicit `unsupported_by_provider` for the current
  single-arm LIBERO provider; no dual-arm capability is fabricated.

The cuRobo methods lazily delegate to the upstream in-process Python module
when its dependencies are installed. There is no invented HTTP endpoint or
`ROBOFORGE_CUROBO_URL` fallback.
