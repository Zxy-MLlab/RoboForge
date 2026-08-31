# Architecture analysis

The receipt remains the authoritative private provenance object. The candidate
does not expose or relocate it. After receipt validation, Core copies only its
boolean result into the already-public AgentEvidence object.

This is less expansive than exposing Adapter sensor reports or independent
verifier prose. It preserves sealed evaluation and evaluator isolation while
making the public evidence contract truthful about its completion eligibility.

The field must be attached after receipt validation and before evidence is
persisted, so immediate results, later inspection, checkpoints, and resumed
model context agree.
