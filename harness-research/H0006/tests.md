# Tests

Implemented candidate regressions covered:

- read -> unchanged same-range replace succeeds without model SHA;
- read -> changed same-range replace fails closed;
- read -> outside-range change -> same-range replace succeeds;
- replace without matching read fails closed;
- different range/path does not reuse a digest;
- explicit low-level H0005 guard remains supported;
- schema no longer asks the model to copy `expected_old_sha256`;
- cache loss/rebuilt AgentLoop fails closed.

Candidate static validation before A/B:

- focused H0006 regressions: 7 passed;
- `tests/test_core_closure.py`: 49 passed;
- full suite: 265 passed, 3 warnings;
- `python -m compileall -q embodied_codex evaluation`: passed;
- `git diff --check`: passed.

Autonomous A/B:

- 16/16 preregistered FakeAdapter runs completed;
- baseline: 11/19 replacements succeeded, 8 WorkspaceErrors, 83 model calls;
- candidate: 12/21 replacements succeeded, 9 WorkspaceErrors, 88 model calls;
- baseline validated successful edits: 5; candidate: 7;
- baseline/candidate verified physical trials: 4/4;
- baseline/candidate valid completions: 0/0.

The safety regressions passed, but the autonomous usable-friction KEEP gate did
not. No LIBERO run was performed.
