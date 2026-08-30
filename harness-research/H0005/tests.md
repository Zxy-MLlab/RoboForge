# Tests

Not written because implementation is intentionally paused.

Required test-first regression:

`test_read_range_digest_can_guard_atomic_line_replacement`

The behavior-level invariant should prove that a digest returned by a read of
an unchanged line range succeeds, while any intervening change or mismatched
range still fails closed. Schema coverage must prove the relationship is
model-visible rather than merely adding an internal field.
