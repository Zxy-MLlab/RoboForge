# Completion audit

This audit intentionally distinguishes implementation from real-world proof.

| Requirement | Current authoritative evidence | Status |
|---|---|---|
| OpenHands is the generic harness | `roboforge.cli`, OpenHands SDK persisted conversations, 19 integration tests | implemented and exercised |
| No second generic AgentLoop in new runtime | static boundary test; no `AgentLoop` import under `roboforge/` | proven for formal entry |
| Real LIBERO observation | `openhands-native-task0-r4`, diagnostic evidence and four sensor artifacts | proven |
| Real Controller execution | task0-r4 and task1-reuse physical ledgers | proven |
| Multimodal evidence | RGB, depth, proprioception and outcome images in immutable CAS | proven |
| Evidence-driven code evolution | Task 8 trials 1-6, Controller snapshots/diffs and authentic success | proven |
| Closed-loop Controller API | public `observe/act/use/verify/record` runtime contract; successful Task 8 trajectory | proven |
| Persistent assets | real partial Experience under `openhands-native-assets` | proven partial |
| Autonomous cross-task retrieval | Task 8 search/read, `assets_used`, Controller evolution and authentic success | proven |
| Capability acquisition | hardened validation/materialization tests and Task 6 physical evidence | acquisition and physical use proven; autonomous acquisition decision remains incomplete |
| Authentic success | Task 8 `experiment://physical-000006`, receipt-bound Controller and environment identity | proven |
| Safety/provenance/recovery | full regression suite; real idempotency incident retained | proven after fix |

The formal CLI provider handoff was exercised by a fresh real campaign
(`openhands-native-task8-autonomous-r2`). The initial launch correctly failed
closed before any physical action when a custom endpoint lacked an explicit
provider; the CLI now forwards `verifier_provider`, `verifier_base_url`, and
`verifier_model` to the Python 3.11 Adapter worker. The relaunch completed four
authentic LIBERO physical trials and then terminated on repeated upstream model
disconnects. This is classified E (provider/system), not task success. The
campaign's immutable ledger and OpenHands conversation remain available for
forensics.

The same campaign was subsequently resumed through its full eight-trial
physical budget. Trial 6 exposed an H-class authentic-verification lifecycle
bug: `begin_execution()` captured the before view, but the immediately
following physical reset cleared it, so the independent outcome verifier
received `before=None`. The LIBERO Adapter now captures the canonical before
view after reset, at the actual S0 state. Trials 7 and 8 prove the repair with
distinct immutable before/after image hashes and no verifier diagnostic. Both
receipts remain authentically false. They also bind the prior Experience
`experience://96b240ff7e09ddc3d4c4c9762ab2da33aea5f97a921802f694b4203da139a16d`
in `assets_used`, proving continued cross-task execution provenance but not
successful transfer.

During resume the Agent initially claimed completion from attachment and
geometric support evidence despite the false task-level receipt. Resume
guidance now explicitly preserves the architectural acceptance boundary:
partial evidence cannot override `physical_verification.verified=false`.
This changes no manipulation strategy and adds no task-specific policy.

Task 2 (`openhands-native-task2-success-r1`) then completed a fresh eight-trial
campaign using only the public instruction “pick up the black bowl from table
center and place it on the plate.” Every trial had valid immutable before/after
RGB evidence after the reset-lifecycle fix, and every trial bound the prior
Experience in `assets_used`. Controller digests changed across all eight trials,
with factual inspection and comparison between revisions. All authentic task
receipts remained false, so this is evidence-driven evolution and transfer,
not success. The Agent independently searched capabilities and invoked
`acquire_capability` twice after identifying a perceived gap; both attempts
were safely rejected because the requested source was outside the confined
OpenHands workspace. This proves the safety boundary and an agent capability
decision attempt, but not the required successful autonomous acquisition,
validation, integration, and physical use chain.

Task 8 (`openhands-native-task8-capability-r1`) provides an additional
capability-transfer trajectory. The Agent searched the persisted capability
catalog, selected and read the existing point-reference extractor, and used
the resulting utility in Controller revisions across six committed real
LIBERO trials. Each trial also carried the prior Experience ID in
`assets_used` and had valid immutable before/after evidence. All receipts were
authentically false. This proves autonomous capability selection and real use,
but not creation of a new capability through the full acquisition chain.

Authoritative success evidence: in the same Task 8 campaign,
`experiment://physical-000006` carries `physical_verification.verified=true`
with Controller digest
`a4174774d88b3db9e6dde78a206a93ef3d49937a4422d88cfa53320dfd956daa`. The
receipt is bound to the LIBERO episode/environment identity and the immutable
before/after RGB artifacts. The Agent's Controller used public RGB-D
perception, point-reference control, calibrated approach/lift/release motions,
attachment verification, and support-relation verification. This is an
authentic full-task success for the instruction “pick up the black bowl next to
the plate and place it on the plate”; no privileged simulator state, verifier
shortcut, task-name branch, or modified reward was introduced. The successful
trial also declares the previously persisted Experience in `assets_used`, so
success and strict cross-task reuse coexist in one real trajectory.

The next acquisition-focused Task 9 launch (`openhands-native-task9-acquisition-r1`)
confirmed the widened workspace editor boundary but was terminated by repeated
external model-provider disconnects before its first physical trial. Its ledger
contains one committed diagnostic and no pending physical reservation. This is
classified E and does not count toward autonomous new-capability acquisition.
Its durable resume later completed one physical trial after autonomously
reading and materializing the existing point-reference capability; that trial
binds both the Capability and prior Experience in `assets_used`. Repeated
provider disconnects then exhausted retries. This remains persisted-capability
reuse, not new acquisition.

An empty-library campaign (`openhands-native-empty-acquisition-task0-r1`) was
also run against the public Task 0 instruction with no pre-existing assets.
The Agent completed two authentic physical trials and iterated its Controller
without fabricating a capability need; the run then ended on repeated external
provider disconnects. The empty asset root contains no registered capability,
so this is honest negative evidence for autonomous new-capability acquisition,
not a missing persistence record.

A durable resume of that empty-library conversation was attempted with twelve
OpenHands provider retries. Every retry ended with the same remote disconnect
before another Agent decision; the physical ledger remained at two committed
trials with no pending reservation and no capability source or asset. Together
with the repeatedly exhausted Task 9 resumes, this is a persistent E-class
external blocker for the sole remaining acceptance proof. The implementation,
authentic success trajectory, and regression evidence remain valid; autonomous
new-capability creation in a real trajectory remains unproven.

The Task B resume incident is classified H: a new tool-call ID bypassed a
pending physical reservation. The committed and pending records are retained.
The service now fails closed across all request IDs while any physical action is
unresolved. It is not counted as reuse or improvement.

Task C provides a factual Controller evolution report via
`compare_trials(physical-000001, physical-000002)`: the second Controller adds
point-reference extraction, physical actions, a fresh observation, attachment
verification and support-relation verification. Both trials remain
authentically unverified; this is behavioral evidence, not a success claim.

Task D (`openhands-native-task3-strict-reuse`) is a second real LIBERO failure
trajectory. The Agent performed observation and one physical RGB-D trial, but
did not read the prior Experience before that trial and produced unreachable
post-return action code. This is classified R/K (reasoning/control), not a
Harness success or verifier shortcut; the immutable evidence and Controller
digest remain available for debugging.

The acquired generic capability `capability://07216908e961c612508c1008fbcc25cec35f8919a782adceb76699fbf70a3515`
was validated with the hardened sandbox and persisted with source digest
`8d9ad41bdfc3ee064a2f2a97d967b8af898cc61ce3b96d9fe3e027e652c7eb29`. Its
Agent-facing materialization path is tested end to end. Task E did not use it:
the Agent's semantic query did not initially match its metadata, and its physical
request was left unresolved/pending. This is recorded as missing capability
transfer, not inferred reuse.

Task 6 provides a separate physical-use proof for the acquired capability:
`capability://07216908e961c612508c1008fbcc25cec35f8919a782adceb76699fbf70a3515`
was read in a session-scoped audit, materialized with its pinned source digest,
imported by `controller.py`, and executed by the real LIBERO Adapter as
`experiment://physical-000001` with Controller digest
`5665f8256280720cf9d2c14259718d9faa14f1b2de70de0394f29769d199b8a5`. The
physical receipt is authentically unverified, so this proves capability
integration/use, not task success.

Task 7 (`openhands-native-task7-success-r2`) is the strongest within-task
evidence so far: four committed physical trials with Controller digests
`335766…9aa1`, `0b425…07cb`, `5c8c…f94e`, and `d5100c…9a63`. The Agent changed
the Controller after each factual outcome, progressing from RGB-D perception to
reference-based control, calibrated grasp proposals, and explicit attachment /
support verification. All four receipts are authentically unverified. The
campaign therefore proves closed-loop trial evolution, not task completion.

Task F (`openhands-native-task5-capability`) proves strict cross-task Experience
reuse. Before its first physical trial the Agent autonomously searched, selected
and read `experience://96b240…a16d`; its `run_controller` action explicitly
declared that ID in `assets_used`. The immutable evidence
`experiment://physical-000001` binds the asset, Controller digest
`b2bf5268c9d4c8e84e3fd6c0c2dda1ecd8043f400d9b9acdc8595d612358a78f`, and
real LIBERO execution. The receipt is authentically unverified, so reuse is
proven while task improvement/success is not.
