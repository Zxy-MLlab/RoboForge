# Results

The generic regression reproduced the field failure on the frozen baseline and
passed after the one-guard fix.

- Focused regression: `1 failed` on baseline, then passed after the fix.
- Related evidence/Decision/diagnostic suite: `46 passed`.
- Full test suite: `248 passed`, 3 third-party deprecation warnings.
- `python -m compileall -q embodied_codex evaluation`: passed.
- `git diff --check`: passed.

Cross-environment embodied validation remains pending because this is a narrow
EventStore correctness fix and only LIBERO is deployed. The generic FakeAdapter
reproduction demonstrates that the defect is independent of task semantics.

## H0004 cumulative cross-task validation

The cumulative baseline/candidate LIBERO comparison is preserved at
`/root/autodl-tmp/harness-validation-H0004/comparison.md`. None of the four
valid runs called `inspect_execution`, so this experiment supplies no direct
embodied validation for H0001. It also exposes no contrary routing result.

Decision: **KEEP** based on the independently reproduced A-class defect, exact
generic regression, and unchanged digest/path validation. Cross-environment
embodied validation remains pending.
