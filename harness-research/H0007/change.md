# Change

Candidate `4a90a557f25aa3e472a3be0dff27cd89a5b9d222` implemented:

- define one reusable Decision-context JSON schema;
- compose it into Agent-facing schemas for consequential Tools;
- atomically record supplied context during dispatch before handler invocation;
- link the exact Tool call and evidence through existing provenance machinery;
- remove the standalone `record_decision` prerequisite from the candidate
  model-facing surface while retaining internal/backward-compatible inspection
  and recovery behavior where required.

No Decision field may be invented or defaulted by Harness.

The candidate passed generic safety/provenance tests but was reverted after
the autonomous A/B. It removed all standalone Decision turns, yet candidate
trajectories produced materially more failed bounded edits and lower
consequential success conversion.
