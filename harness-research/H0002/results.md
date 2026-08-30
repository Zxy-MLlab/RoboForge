# Results

The generic contract regression fails before the change and passes after it.
Model integration and autonomy-boundary tests remain green.

Two paired autonomous FakeAdapter runs used Apex `gpt-5.6-sol`, high reasoning,
the same 12-step/2-trial budgets, equivalent public marker tasks, and identical
Tool availability. The only Harness variable was the description suffix.

| condition | Tool calls | missing-Decision rejections | total Tool errors | physical trials |
|---|---:|---:|---:|---:|
| H0001 baseline (2 runs) | 29 | 5 (17.2%) | 12 | 1 |
| H0002 candidate (2 runs) | 23 | 1 (4.3%) | 9 | 2 |

The first pair was noisy: the candidate spent several calls submitting invalid
evidence references and executed no physical trial. The second candidate run
had zero missing-Decision rejections and used both physical trials. Therefore
the paired evidence supports the primary protocol-friction metric but does not
support a task-success claim.

Experiment artifacts are under `/root/autodl-tmp/h0002-experiment/`.
Cross-environment validation remains pending.

Validation:

- focused H0001/H0002 regressions: passed;
- model integration/autonomy boundary suite: 25 passed;
- full suite: 249 passed, 3 third-party deprecation warnings;
- `python -m compileall -q embodied_codex evaluation`: passed;
- `git diff --check`: passed.

## H0004 cumulative cross-task validation

Across the four valid controlled LIBERO runs, the frozen baseline produced five
missing-Decision rejections in 44 Tool calls: three on Task 1 and two on Task 2.
The cumulative Candidate produced zero in 36 Tool calls across the same two
tasks. Overall ProtocolError results changed from 8/44 to 0/36.

This supports the intended generic protocol-friction metric across more than
one LIBERO task. It does not support a task-success claim. Decision: **KEEP**.
Cross-environment validation remains pending.
