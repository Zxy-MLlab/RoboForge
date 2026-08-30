# Tests

Regression:

`test_consequential_tool_schema_declares_decision_prerequisite`

It verifies that a generic mutating Tool exposes its consequence and
`record_decision`, while a generic read-only Tool does not claim the
prerequisite.

Baseline: failed because the description was `Mutate generic state.` only.

Post-change:

- focused H0001/H0002 regressions: 2 passed;
- model integration and autonomy-boundary tests: 25 passed.

Primary future experiment metric: Decision-prerequisite rejection rate.
Secondary metrics: valid action conversion and model calls per physical trial.
