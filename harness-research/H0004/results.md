# Results

One paired autonomous FakeAdapter run held model, task, budgets, and Tool
availability constant.

| condition | Decision attempts | accepted | reference-format rejections | physical trials |
|---|---:|---:|---:|---:|
| H0003 baseline | 6 | 4 (66.7%) | 2 | 1 |
| H0004 candidate | 4 | 4 (100%) | 0 | 1 |

The paired result supports improved valid Decision conversion. It does not
establish task-performance improvement. Artifacts are under
`/root/autodl-tmp/h0004-experiment/`.

Validation:

- focused Decision contract/rejection tests: passed;
- full suite: 251 passed, 3 third-party deprecation warnings;
- `python -m compileall -q embodied_codex evaluation`: passed;
- `git diff --check`: passed.

Cross-environment validation remains pending.

## Cumulative cross-task validation

Protocol and full metrics:

- `/root/autodl-tmp/harness-validation-H0004/protocol.json`
- `/root/autodl-tmp/harness-validation-H0004/metrics.json`
- `/root/autodl-tmp/harness-validation-H0004/comparison.md`
- `/root/autodl-tmp/harness-validation-H0004/excluded-runs.md`

Four valid runs compared frozen baseline `ce260a73` and cumulative Candidate
`fe9506f` on LIBERO Tasks 1 and 2 with 20 Agent steps and two physical trials.
The baseline had three invalid evidence-reference Decision rejections; the
Candidate had zero. Accepted Decision conversion changed from 4/8 to 10/10.

One initial Task 2 baseline was excluded as B because Apex timed out on request
7 and no terminal result was produced; the complete retry is the only Task 2
baseline pooled. No run completed its embodied task.

Decision: **KEEP**. The result supports truthful routing-schema conversion, not
task performance. Cross-environment validation remains pending.

Post-comparison Candidate verification:

- four H0001-H0004 focused regressions: `4 passed`;
- full `tests evaluation` suite: `251 passed`, 3 third-party deprecation warnings;
- `python -m compileall -q embodied_codex evaluation`: passed;
- `git diff --check`: passed.
