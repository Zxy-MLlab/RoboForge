# KEEP / REVERT

## Decision

REVERT H0006. KEEP H0005.

## Gate evaluation

- Safety gates: PASS. No unread or stale selected range was overwritten;
  outside-range changes remained mergeable; cache loss failed closed; the
  explicit H0005 low-level API remained intact.
- Task independence: PASS. FakeAdapter-only implementation and tests contained
  no embodied or LIBERO strategy.
- Autonomous valid-intent conversion: FAIL. Replace success was 57.9% baseline
  versus 57.1% candidate; WorkspaceError rate was 42.1% versus 42.9%.
- Calls/errors: FAIL. Candidate required 88 model calls versus 83 and produced
  9 WorkspaceErrors versus 8.
- End-to-end outcome: NO IMPROVEMENT. Both conditions produced four verified
  physical trials and zero valid completed runs.

The candidate removed manual SHA carriage from the schema but introduced an
equally model-visible exact-range-read prerequisite. That is not the intended
low-friction interaction surface. Production code is therefore reverted and
LIBERO is not run for H0006.
