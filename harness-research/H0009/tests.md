# Tests

Stage 1 generic regressions:

- false authentic receipt is visible as public false status;
- true authentic receipt is visible as public true status;
- Controller-local verification may disagree without changing the status;
- `inspect_execution` roundtrips the status;
- latest model context roundtrips the status;
- diagnostics never expose physical verification;
- receipt metadata and environment identity do not leak;
- `finish` behavior remains receipt-gated.

Stage 2 autonomous FakeAdapter A/B measures false `finish` attempts, unchanged
physical retries, calls before corrective edit, valid completion, and Tool
errors.

## Results so far

- Baseline focused regression: `5 failed, 2 passed`; failures were the
  preregistered missing public status, not fixture or protocol errors.
- Candidate focused regression: `7 passed`.
- Existing core/post-closure/correctness closure set: `73 passed`.
- Autonomous A/B: not started.
