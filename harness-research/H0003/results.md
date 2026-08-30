# Results

One paired autonomous FakeAdapter run used the same model, task, budgets, and
Tool availability. The only Harness variable was the Controller entrypoint
text.

| condition | diagnostic attempts | completed diagnostics | physical trials |
|---|---:|---:|---:|
| H0002 baseline | 2 | 0 | 0 |
| H0003 candidate | 1 | 1 | 1 |

The baseline first called diagnostic with no file and later produced a source
without `run(robot)`. The candidate wrote the declared entrypoint, committed a
completed read-only diagnostic, and proceeded to one physical trial. Neither
condition solved the generic marker task; no task-success benefit is claimed.

Artifacts: `/root/autodl-tmp/h0003-experiment/`.

Validation:

- focused H0001-H0003 regressions: 3 passed;
- related diagnostic/model integration suite: 31 passed;
- full suite: 250 passed, 3 third-party deprecation warnings;
- `python -m compileall -q embodied_codex evaluation`: passed;
- `git diff --check`: passed.

Cross-environment validation remains pending.

## H0004 cumulative cross-task validation

On LIBERO Task 1, the frozen baseline attempted one diagnostic before creating
the Controller and later consumed its physical trial with
`controller must define run(robot)`. The cumulative Candidate committed two
diagnostic evidence objects and then ran an action-bearing physical Controller
for 158 Controller steps. This is a direct code-to-execution conversion result.

Task 2 did not reproduce the same contrast: its baseline already used a valid
entrypoint, while the Candidate did not attempt a diagnostic and its two
physical Controllers stopped on model-authored capability input errors. Those
errors are outside H0003's source/entrypoint contract and do not show an H0003
regression.

Decision: **KEEP**, with cross-task benefit still incomplete and
cross-environment validation pending. No task-performance claim is made.
