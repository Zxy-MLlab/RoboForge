# Change

Not implemented.

Candidate minimal change:

- define one reusable Decision-context JSON schema;
- compose it into Agent-facing schemas for consequential Tools;
- atomically record supplied context during dispatch before handler invocation;
- link the exact Tool call and evidence through existing provenance machinery;
- remove the standalone `record_decision` prerequisite from the candidate
  model-facing surface while retaining internal/backward-compatible inspection
  and recovery behavior where required.

No Decision field may be invented or defaulted by Harness.
