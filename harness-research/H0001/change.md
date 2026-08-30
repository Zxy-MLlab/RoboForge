# Change

`AgentLoop._inspect_execution()` now ignores EventStore rows whose kind is not
`execution` before loading evidence references.

The change matches `_execution_by_ref()`, which already used this authoritative
event-kind filter. SHA verification, path isolation, public projection, and
unknown-reference failure behavior are unchanged.

No task knowledge, task policy, capability preference, or retry behavior was
added.
