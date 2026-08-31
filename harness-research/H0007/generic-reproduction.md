# Generic reproduction

Baseline FakeAdapter behavior:

1. Model emits one valid `run_controller` request containing complete explicit
   Decision context.
2. Baseline schema rejects the extra context or the dispatch protocol rejects
   execution because no prior `record_decision` established pending state.
3. The model must spend another response on `record_decision`, then another on
   `run_controller`.

Candidate behavior:

1. The same one-response intent validates.
2. Harness durably records the supplied context.
3. Harness links the Tool invocation and resulting evidence to that record.
4. The physical trial executes exactly once.

Fail-closed controls:

- missing Decision context rejects before mutation;
- malformed context rejects before mutation;
- invalid evidence references reject before mutation;
- EventStore failure rejects before mutation;
- unknown/unrecognized consequence remains rejected;
- READ_ONLY and VALIDATION calls require no Decision context;
- process resume never fabricates or reuses unrelated pending context;
- duplicate Tool delivery does not execute a physical trial twice.
