# KEEP / REVERT

## Decision: KEEP candidate for controlled real validation

The candidate passed the preregistered generic and safety gates:

- focused generic regressions: 7/7 passed;
- affected closure tests: 73/73 passed;
- authentic receipt metadata remains private;
- diagnostic evidence has no physical-completion field;
- `finish` remains bound to the authentic private receipt;
- corrected autonomous disagreement A/B reduced model calls from 55 to 24,
  physical trials from 11 to 4, and raised valid completion from 1/4 to 4/4;
- all four candidate disagreement runs preserved the already-correct
  Controller, versus seven changed-Controller executions on baseline;
- Tool and protocol errors remained zero;
- the control condition remained 4/4 valid completion with 29 versus 30 model
  calls.

The result is task-, strategy-, model-, and environment-independent. It exposes
one already-validated factual boolean and makes no action, Tool, retry, or
completion decision for the model.

KEEP here authorizes preregistration of matched real LIBERO validation. It does
not make task success sufficient evidence and does not authorize changing the
candidate during a frozen run.
