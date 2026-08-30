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
