# KEEP / REVERT

## Decision

REVERT H0007. KEEP retained H0005 behavior.

## Gate evaluation

- Safety/provenance: PASS. Missing/malformed context, bad evidence references,
  EventStore failure, and duplicate physical delivery failed closed. All
  candidate calls/evidence had exact Decision linkage.
- Task independence: PASS. FakeAdapter-only code/tests contained no embodied
  strategy.
- Decision-only overhead: PASS. Standalone Decision calls fell from 27 to 0.
- Model-call efficiency: SMALL IMPROVEMENT. Total calls fell from 90 to 86;
  calls per consequential success fell from 3.91 to 3.44.
- Autonomous valid-intent conversion: FAIL. Success rate fell from 88.5% to
  69.4%.
- Error gate: FAIL. Tool errors rose from 3 to 12; WorkspaceErrors rose from 3
  to 11.

The candidate demonstrated that atomic provenance is technically sound, but
did not demonstrate a lower-friction autonomous surface without meaningful
regression. Production code is reverted and LIBERO is not run.
