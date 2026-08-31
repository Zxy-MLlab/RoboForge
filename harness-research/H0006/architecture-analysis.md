# Architecture analysis

Candidate boundary: Agent-facing Tool wrappers created by `AgentLoop`, not the
general `PersistentWorkspace` API.

The wrapper records only data already returned publicly: normalized path/range
identity and `range_sha256`. Replacement calls pass the stored digest into the
unchanged H0005 check. The cache is execution-session bookkeeping, not task
state or semantic reasoning.

No-cache behavior must reject and request a fresh read. This makes crash/resume
safe without persisting or reconstructing an opaque token cache. Programmatic
legacy callers can continue to pass an explicit digest or use the existing
optional API directly.
