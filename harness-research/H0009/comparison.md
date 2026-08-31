# Comparison

## Autonomous FakeAdapter A/B

The retained H0008 baseline `2f6c899c83c88d18c3eeef70d3c5cc04fe1345c5`
and H0009 candidate code commit `5ef4c242f4bc5c1867538d92bcae9f03a5e48664`
each ran eight valid times: two generic disagreement conditions, four
replicates, Apex `gpt-5.6-sol` high, 16 Agent steps, and three physical trials.
The prompt did not name or explain `physical_verification`.

| Metric | Baseline | Candidate |
|---|---:|---:|
| Model calls | 85 | 53 |
| Tool calls | 89 | 53 |
| Tool errors | 0 | 0 |
| Protocol errors | 0 | 0 |
| Physical trials | 15 | 9 |
| Valid completions | 5/8 | 8/8 |

The discriminating condition seeded a correct Controller while the
Controller-local verifier returned false and the independent authentic receipt
returned true.

| Disagreement metric | Baseline | Candidate |
|---|---:|---:|
| Model calls | 55 | 24 |
| Physical trials | 11 | 4 |
| Valid completions | 1/4 | 4/4 |
| Executions of changed Controller | 7 | 0 |
| Tool/protocol errors | 0 | 0 |

All four candidate runs observed the exact public combination
`local_verified=false`, `physical_verification.verified=true`, then finished
after one physical trial without modifying the correct Controller. Baseline
could not observe authentic eligibility, modified the correct Controller in
all four runs, and completed only once.

The control condition seeded an incorrect Controller while its local verifier
could report true. Both conditions corrected the source before successful
completion. Valid completion remained 4/4 in each condition; candidate model
calls were 29 versus 30, Tool errors remained zero, and no erroneous finish
after a false receipt occurred. One candidate control run used a second trial
after a false authentic receipt and changed the Controller before retrying.

No public evidence contained receipt identity, environment identity, resume
token, or episode identity.

## Invalid infrastructure attempts

The first version of the disagreement fixture accidentally let FakeAdapter's
default receipt depend on the Controller-local verifier. Its runs were rejected
before analysis. A direct fixture check then proved the corrected invariant:
`local_verified=false`, `authentic_receipt_verified=true`, and
`execution.completed=true` for the same execution.

Four additional attempts produced no result because of Apex connection
disconnects or operator interruption. They remain preserved and are excluded
mechanically rather than overwritten. Raw runs and the fixed valid-run manifest
are under `/root/autodl-tmp/harness-validation-H0009/`; aggregate metrics are in
`ab-metrics.json`.

## Matched real LIBERO validation

H0009 used the same task order `[3, 4, 7]`, state `0`, model/provider,
reasoning settings, capability snapshot, budgets, sandbox, and reset protocol as
H0008. The candidate run used 951 model calls, 1,037 Tool calls, 101 failed
Tool calls, 73 physical trials, and completed task 4 only. H0008 used 675, 718,
63, 65, and also completed one task. Per-task comparison:

| Task | H0008 steps/trials/result | H0009 steps/trials/result |
|---:|---|---|
| 3 | 228 / 30 / false | 359 / 30 / false |
| 4 | 373 / 30 / false | 126 / 13 / true |
| 7 | 74 / 5 / true | 466 / 30 / false |

This is a mixed real-world result, not a general score improvement. The
candidate's factual projection was mechanically correct in all 73 physical
evidence objects and did not alter diagnostics or private completion gates.
The campaign produced no evidence of a generic safety or provenance regression.
