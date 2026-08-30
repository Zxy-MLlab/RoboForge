# Tests

Generic regression:

`test_inspect_later_execution_ignores_non_evidence_decision_links`

Baseline result: failed with the P30 error at `_load_evidence_reference()`.

Post-change focused result: 2 passed.

Post-change related result: 46 passed.

Expected measurable effect: valid evidence inspection success increases while
protocol error rate decreases; evidence digest validation remains unchanged.
