# Tests

Implemented generic regressions covered:

- one-response Decision + physical operation succeeds and links evidence;
- one-response Decision + workspace mutation succeeds and links provenance;
- missing context fails before handler invocation;
- malformed context fails before handler invocation;
- invalid evidence URI fails before handler invocation;
- Decision EventStore failure prevents mutation;
- read-only and validation Tools do not require Decision context;
- Decision identifier is Harness-generated from the Tool call identity;
- replay/recovery cannot double-execute a physical operation;
- checkpoint resume cannot reuse unrelated Decision context;
- all consequential Agent schemas require the same envelope;
- no standalone Decision prerequisite remains in the candidate Agent surface;
- existing Decision/evidence audit and listing remain intact.

Candidate static validation:

- H0007 focused/core tests: 53 passed;
- affected compatibility set: 98 passed;
- full suite: 269 passed, 3 warnings;
- `python -m compileall -q embodied_codex evaluation`: passed;
- `git diff --check`: passed.

Autonomous FakeAdapter A/B:

- 16/16 preregistered runs completed;
- baseline: 90 model calls, 27 Decision-only calls, 23/26 successful
  consequential operations, 3 Tool errors;
- candidate: 86 model calls, zero Decision-only calls, 25/36 successful
  consequential operations, 12 Tool errors;
- all 36 candidate consequential calls had exact durable Decision provenance;
- all 12 candidate physical executions had exact evidence links.

Safety/provenance passed. Autonomous conversion/error KEEP gates failed. No
LIBERO run was performed.
