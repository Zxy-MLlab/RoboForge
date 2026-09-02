# RoboForge final validation report

## Architecture and ownership

The canonical path is `OpenHands Agent -> persistent workspace -> RoboForge
Control Plane -> LIBERO Runtime Provider -> LIBERO`. OpenHands SDK owns the only
generic agent loop, conversation, editor, terminal, planning and subagent tools.
`roboforge/runtime.py` only binds embodied operations. Candidate Controllers see
the Robot SDK, never reset/seed/snapshot controls, simulator internals, the sealed
success evaluator, or promotion authority. `roboforge/service.py` freezes source,
accounts trials and commits immutable evidence. `roboforge/control_plane.py` owns
environment inspection, evidence-only replay, paired comparison and promotion.
The registry in `roboforge/assets.py` is the single capability metadata source;
its terminal promotion/rejection decision is immutable.

The Python 3.12 OpenHands process and Python 3.11 LIBERO process communicate over
an authenticated mode-0600 Unix socket. The provider owns reset, sensor/action
access, safety and verification. Evaluation runs outside both Agent and candidate.

## Upstream mapping and licenses

| Upstream | Revision inspected | License | Mechanism and RoboForge location |
| --- | --- | --- | --- |
| OpenHands SDK | `704cbe6015e3d59cabe04632175d99df2d448999` / installed `1.44.1` | MIT | Sole agent/conversation/tool substrate in `roboforge/cli.py`, `runtime.py` |
| NVlabs/ASPIRE | `f4c8939aab0af9b97690c561bd80e282940f7886` | Apache-2.0 plus third-party notices | Primitive trace design from `aspire/sim/cap/integrations/libero_trace_logger.py`; trial/artifact lifecycle from `aspire/sim/cap/envs/trial.py`, represented by provider/service evidence |
| capgym/CaP-X | `53e9966d7a8e2fa7494676772bccc35280f5c0ed` | repository license reviewed | Code-as-policy LIBERO API boundary in `embodied_codex/deployments/libero.py` and the bridge; no coordinator copied |
| NVIDIA ENPIRE | public project page inspected 2026-09-02 | publication/project terms; no code copied | External reset, execute, auto-evaluate, refine ownership in `service.py` and external evaluation scripts |
| KE7/HELIX | `858b6bcbafd9bb1ca9226e1f03c83d8cbe3a0db6` | BSD-3-Clause | Fair parent/candidate comparison, informed by `src/helix/eval_policy.py`, in `scripts/run_paired_libero_evaluation.py` |

No ASPIRE fixed Coordinator/Actor/fix loop or HELIX training loop was copied.

## Migration and deprecation

`embodied_codex/kernel` remains compatibility and reusable infrastructure, but is
not the canonical agent core. Its generic AgentLoop, provider loop, context
builder, workspace and special capability-development tools are deprecated for
new runs. `roboforge-openhands`/`roboforge` is the sole canonical entry path.
Robot primitives remain Controller APIs and are not registered as OpenHands tools.

## Real LIBERO validation

All results below used the RTX 5090 deployment and the real LIBERO environment,
not `FakeAdapter`.

Runtime consistency used public task 0, state 0, and the identical frozen
`validation/controllers/runtime_consistency.py`. Direct LIBERO and RoboForge had
equal task identity and instruction, both ended at step 12, and produced matching
trace semantics. Evidence:
`/root/autodl-tmp/experiments/goal-runtime-consistency-20260902-r2`.
The earlier metadata-failed attempt is preserved at
`/root/autodl-tmp/experiments/goal-runtime-consistency-20260902`.

Autonomous development used LIBERO task 8 and development state 0. The campaign
is `/root/autodl-tmp/experiments/openhands-native-task8-capability-r1`. Initial
Controller SHA256 was `0b2b96e75b59583723f000fe7987341e3457db72a3c8d0de9a2e6b82ffa02b32`;
the frozen final SHA256 was
`a4174774d88b3db9e6dde78a206a93ef3d49937a4422d88cfa53320dfd956daa`.
Independent paired evaluation sealed Controller I/O before calling
`env.check_success()`. Across held-out states 1-49, initial was 0/49 (0%) and
final was 13/49 (26.53%). The final was frozen before held-out evaluation and was
not retuned. Records:

- states 1-5: `/root/autodl-tmp/experiments/goal-paired-task8-initial-final-heldout-1-5-20260902`
- states 6-49: `/root/autodl-tmp/experiments/goal-paired-task8-initial-final-heldout-6-49-20260902`
- immediate-previous versus final negative result (1/5 tie): `/root/autodl-tmp/experiments/goal-paired-task8-heldout-1-5-20260902`

Reviewable lightweight copies are committed under
`validation/evidence/task8_paired_evaluation/`. `paired_results.json` and
`paired_results.csv` contain both arms for every state 1-49; the accompanying
Controller snapshots, manifests and `checksums.sha256` make the reviewed inputs
independently checkable without committing simulator caches or videos.

Capability `capability://07216908e961c612508c1008fbcc25cec35f8919a782adceb76699fbf70a3515`
was promoted externally from authentic task-8 physical evidence. A fresh task-0
Controller materialized and used it without execution error; the ID is bound in
trial provenance at
`/root/autodl-tmp/experiments/goal-promoted-capability-reuse-task0-20260902`.
The compact promotion and reuse records are committed under
`validation/evidence/capability_reuse/`. Runtime consistency's compact manifest
and result are under `validation/evidence/runtime_consistency/`.
The ordinary `roboforge run` path also completed task 0 at
`/root/autodl-tmp/experiments/goal-cli-run-task0-20260902`.

## Reproduction

```bash
python -m pip install -e '.[test,openhands]'
source /root/autodl-tmp/roboforge_libero_env.sh

python scripts/validate_runtime_consistency.py --task 0 --state 0 \
  --output /new/path/runtime-consistency
python scripts/run_paired_libero_evaluation.py --task 8 --states 1-49 \
  --baseline /path/to/initial.py --candidate /path/to/final.py \
  --output /new/path/paired
python scripts/validate_promoted_capability_reuse.py \
  --asset-root /root/autodl-tmp/experiments/openhands-native-assets \
  --capability capability://07216908e961c612508c1008fbcc25cec35f8919a782adceb76699fbf70a3515 \
  --task 0 --state 0 --output /new/path/reuse

python -m pytest -q tests evaluation
OPENHANDS_SUPPRESS_BANNER=1 /root/autodl-tmp/RoboForge-v2-spike/.venv/bin/python \
  -m pytest -q tests/test_openhands_native.py
python -m compileall -q roboforge embodied_codex evaluation scripts
git diff --check
```

## Accounting and limitations

The held-out comparison contains 98 real episodes (49 paired states), plus two
runtime-consistency arms, development trials, one capability-reuse trial and one
canonical CLI trial. The two paired manifests record about 21.9 minutes wall
clock in total; all real simulation ran on one RTX 5090. Development model token
usage was not emitted reliably by the configured provider, so an exact token
total cannot be reconstructed and is not fabricated. This is an accounting
limitation. Development used state 0 rather than the requested numeric 51-65
labels because this LIBERO task exposes indexed initial states; held-out indices
1-49 remained sealed and disjoint. The 26.53% final success rate is a clear
paired improvement, not universal task mastery. Failures and the prior/final tie
are retained as negative evidence. No claim here treats mocks as physical proof.
