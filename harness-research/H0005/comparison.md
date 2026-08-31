# Comparison

The deterministic comparison is conclusive: the baseline cannot obtain the
authoritative replacement-range digest from `read_file`; H0005 can, and the
returned digest succeeds while an intervening same-range change fails closed.

The eight-run autonomous FakeAdapter A/B is **inconclusive** for model-level
friction. Candidate runs made two guarded replacement attempts and both
succeeded; baseline runs made zero guarded attempts, so there is no baseline
success-rate denominator. No causal task-score or call-count improvement is
claimed.

The frozen three-task LIBERO follow-up is observational:

- 22/22 replacements using the exact most-recent same-range digest succeeded;
- 27/27 replacements using another digest failed closed;
- one malformed request failed for an unrelated path error;
- all 60 physical trials completed without Harness/provider failure.

This validates H0005 semantics and shows residual manual-token friction. It is
not a matched baseline/candidate LIBERO comparison.
