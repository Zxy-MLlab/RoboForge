# Comparison

## Autonomous FakeAdapter A/B

Retained H0005 baseline `368d704f52ebb9ee685f215494a25f4784da0e0f` and
H0007 candidate `4a90a557f25aa3e472a3be0dff27cd89a5b9d222` each ran
eight times: two generic tasks, four repetitions, Apex `gpt-5.6-sol` high, 16
Agent steps, two physical trials, and prompts that did not explain Decision
protocol.

| Metric | H0005 | H0007 |
|---|---:|---:|
| Model calls | 90 | 86 |
| Tool calls | 96 | 93 |
| Decision-only calls | 27 | 0 |
| Consequential success / attempts | 23 / 26 | 25 / 36 |
| Consequential success rate | 88.5% | 69.4% |
| Model calls / consequential success | 3.91 | 3.44 |
| WorkspaceErrors | 3 | 11 |
| All Tool errors | 3 | 12 |
| Physical trials | 12 | 12 |
| Verified physical executions | 8 | 8 |
| Valid completed runs | 2 | 4 |

The candidate eliminated separate Decision rounds, and every inline context
was durably committed and linked before execution. Nevertheless it made ten
more consequential attempts for only two more successes. Tool error rate rose
from 3.1% to 12.9%, and valid-intent conversion fell by 19 percentage points.

Edit completion improved from 2/4 to 4/4. Repeat completion is not comparable:
the required second physical execution exhausted the preregistered two-trial
limit before either condition could spend a later finish turn.

The explicit KEEP gate required friction reduction without more errors. That
gate failed, so H0007 is reverted and no LIBERO validation is authorized.

Full raw results are retained outside the repository at
`/root/autodl-tmp/harness-validation-H0007/`.
