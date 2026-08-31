# Architecture analysis

## Candidate boundary

The candidate belongs at the generic Agent Tool schema/dispatch boundary.
Consequential Kernel Tools receive a required structured Decision envelope in
their Agent-facing schema. Dispatch validates and records that explicit model
payload before invoking the existing handler, then uses the existing Decision
link, EventStore, checkpoint, and evidence paths.

Handlers such as workspace mutation, capability registration, reset, and
physical execution must not learn task policy or parse natural-language model
output. The envelope is routing/provenance metadata, not semantic inference.

## Preserved semantics

- The model remains the sole author of every Decision field.
- Consequence classification remains authoritative in `KernelTool` metadata.
- Missing context remains fail closed.
- Physical trials, reset, evidence, sandbox, and verifier behavior are
  unchanged.
- READ_ONLY/VALIDATION calls remain outside the mutation Decision gate.
- EventStore remains audit/provenance, not a reasoning engine.

## Non-goals

- Do not infer a Decision from task text or Controller code.
- Do not automatically decide to edit, execute, acquire, retry, or search.
- Do not weaken or remove Decision provenance.
- Do not add task- or environment-specific behavior.
