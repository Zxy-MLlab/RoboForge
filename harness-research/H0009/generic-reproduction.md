# Generic reproduction

Before production changes, add a FakeAdapter regression with an authentic
physical receipt that disagrees with a Controller-local public verifier.

Baseline expectations:

- physical AgentEvidence contains the local verifier result;
- physical AgentEvidence does not state authentic verification eligibility;
- `finish` knows the receipt is false, proving Harness already has the fact.

Candidate expectations:

- physical AgentEvidence and `inspect_execution` expose only
  `physical_verification.verified`;
- diagnostic evidence has no such field;
- no receipt metadata or private Adapter fields cross the model boundary;
- completion gate semantics are unchanged.
