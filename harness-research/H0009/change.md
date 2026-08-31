# Change

Candidate implementation:

- after the Adapter receipt passes its existing Controller/environment/type
  binding checks, copy only `receipt["verified"]` into
  `agent_evidence["physical_verification"]`;
- preserve that optional field in bounded evidence summaries only when its
  exact shape is `{verified: bool}`;
- leave diagnostics, the private receipt, `finish`, Adapter contracts, Prompt,
  Decision, reset, Controller, and capability policy unchanged.

The candidate adds no task, environment, strategy, action, or Tool knowledge.
