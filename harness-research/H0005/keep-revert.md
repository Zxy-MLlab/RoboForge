# KEEP / REVERT decision

Decision: **KEEP H0005 for correctness and truthful guarded-edit support.**

Reasons:

- the baseline regression fails and H0005 passes;
- exact same-range digests succeeded 22/22 in the real campaign;
- changed or mismatched selected ranges still fail closed;
- no safety, atomicity, provenance, sandbox, or task policy changed;
- no task-specific behavior entered Core;
- full tests, compileall, and diff checks passed.

Limit of the decision: do not claim H0005 improved LIBERO task success or model
call efficiency. The autonomous FakeAdapter A/B is inconclusive and the real
campaign has no matched baseline. Residual manual SHA friction is a separate,
future H candidate requiring its own preregistration and generic A/B.
