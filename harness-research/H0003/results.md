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
