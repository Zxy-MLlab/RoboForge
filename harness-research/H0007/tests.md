# Tests

Not written yet.

Required generic regressions:

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
