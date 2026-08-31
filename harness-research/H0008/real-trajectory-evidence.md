# H0008 frozen real-trajectory evidence

## Campaign

The preregistered campaign completed under frozen Harness SHA
`d9bd7ce6f816bf2a9dcbae6b7608e7fa8a7c15fa` with Apex
`gpt-5.6-sol`, reasoning effort `high`.

| Task | Agent calls | Diagnostics committed | Physical trials | Result |
|---:|---:|---:|---:|---|
| 3 | 228 | 2 | 30/30 | not completed |
| 4 | 373 | 5 | 30/30 | not completed |
| 7 | 74 | 1 | 5/30 | completed |

All cases used development state 0 and fresh workspaces. The frozen Adapter
supports only `libero_spatial`; this campaign is cross-scene evidence inside
one suite, not cross-suite validation.

## Exact aggregate observations

- 675 model calls, 718 Tool calls, and 65 physical trials.
- 63 Tool calls failed (8.77%).
- 11 diagnostic calls produced 8 durable diagnostic evidence objects.
- 63 guarded replacements produced 32 successful edits and 31
  `WorkspaceError` failures. Every call using the most recent exact returned
  range digest succeeded (32/32); every other call failed (31/31).
- 161 model calls (23.85%) were Decision Records.
- 175 calls (25.93%) were evidence/artifact inspection.
- 18 calls were explicit `reset_case`, although each episodic
  `run_controller` already performs an internal S0 reset.
- 31/65 trials immediately reused the prior Controller SHA. Task 3 accounted
  for 21 unchanged reruns.
- Four physical trials performed no actions (two syntax failures, one runtime
  key error, and one Controller path that returned without actions).
- No task invoked Internet search, fetch/download/build/register/test, or
  acquired a new Tool. There were four `search_assets` and two
  `activate_tool_group` calls.
- There were zero context compactions. Tool descriptions and prior evidence
  were not removed by Harness compaction.

## Cross-task facts

Task 7 shows that the frozen interaction surface can support autonomous
success. Trial 2 verified attachment but not placement; Trials 3 and 4 were
code regressions; Trial 5 used a repaired Controller and received a valid
physical verification receipt.

Task 3 exposes the strongest interaction-surface contradiction. Its
Controller-local public `visual_attachment` and `visual_support_relation`
calls sometimes returned `verified=true`, while the independent sensor-only
before/after outcome still found the original target at the source or the
target absent from the plate. The authoritative physical receipt was therefore
false. The model-visible `AgentEvidence` projection retained local verifier
records and images but omitted the physical receipt's final eligibility bit
and the independent public outcome reason. The Agent then authored Progress
records treating Trials 9 and 10 as working success, attempted `finish` ten
times, and repeatedly reran the same Controller.

Task 4 used 19 distinct Controller SHAs. Twenty-three trials reported every
commanded action reached, but independent task outcome remained false in all
30 trials. This separates low-level motion reach from task completion and
supports a mixed capability/model diagnosis rather than a numeric-control
interface defect.

## Integrity

The read-only audit verified:

- all 1,689 EventStore records and their chained digests;
- all three checkpoint envelopes;
- all 73 execution/diagnostic evidence references;
- all 65 immutable Controller snapshots;
- all 2,096 artifact-manifest entries.

No missing file, malformed JSON, digest mismatch, provider outage, sandbox
failure, Adapter crash, or lost context was found.

Machine-readable evidence and extraction code are stored outside the repo at
`/root/autodl-tmp/experiments/libero-real-20260831-h0008/`.
