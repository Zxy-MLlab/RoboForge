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
- Autonomous A/B: 16/16 valid preregistered runs completed after correcting and
  directly verifying an initially invalid synthetic disagreement fixture.
- Candidate disagreement condition: 24 model calls, 4 physical trials, 4/4
  valid completions, zero changed-Controller executions, zero Tool/protocol
  errors.
- Baseline disagreement condition: 55 model calls, 11 physical trials, 1/4
  valid completions, seven changed-Controller executions, zero Tool/protocol
  errors.
- Control condition: 4/4 valid completions for both; 30 baseline versus 29
  candidate model calls; zero Tool/protocol errors.
- Public metadata leak audit: zero.
