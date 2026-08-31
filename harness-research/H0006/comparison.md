# Comparison

## Autonomous FakeAdapter A/B

H0005 baseline `dc35a0e9af9b55a37d9e9034472a7bfe98670fd0` and H0006 candidate
`1e7b2cbf48e90d80632884c226c8d4d78e552de0` each ran eight times: two generic
editing tasks, four repetitions, Apex `gpt-5.6-sol` high, 12 Agent steps, one
physical trial, and prompts that did not mention hashes or concurrency guards.

| Metric | H0005 | H0006 |
|---|---:|---:|
| Model calls | 83 | 88 |
| Tool calls | 86 | 98 |
| Replace success / attempts | 11 / 19 | 12 / 21 |
| Replace success rate | 57.9% | 57.1% |
| WorkspaceError / replace | 42.1% | 42.9% |
| Validated successful edits | 5 | 7 |
| Model calls / validated successful edit | 16.60 | 12.57 |
| Syntactically valid final Controllers | 6 | 8 |
| Verified physical trials | 4 | 4 |
| Valid completed runs | 0 | 0 |

All 19 baseline replace attempts manually carried a digest. All 21 candidate
attempts naturally omitted it. In every candidate run, however, the Agent first
read a broad range and then edited a narrower range. The exact-range cache
therefore rejected the first edit and required another narrow read plus retry.
One later stale selected-range attempt also correctly failed closed.

H0006 preserved safety but did not improve replacement success, WorkspaceError
rate, total model calls, verified trials, or completion. Better final syntax in
two candidate runs did not produce an end-to-end advantage. The preregistered
KEEP gate failed, so the candidate is reverted and no LIBERO validation is
authorized.

Full raw comparison and metrics are retained outside the repository at
`/root/autodl-tmp/harness-validation-H0006/`.
