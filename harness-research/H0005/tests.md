# Tests

Not written because implementation is intentionally paused.

Required test-first regression:

`test_read_range_digest_can_guard_atomic_line_replacement`

The behavior-level invariant should prove that a digest returned by a read of
an unchanged line range succeeds, while any intervening change or mismatched
range still fails closed. Schema coverage must prove the relationship is
model-visible rather than merely adding an internal field.

Required boundary coverage:

- LF range with a final newline;
- last line without a final newline;
- CRLF input using the replacement path's authoritative normalization;
- selected range unchanged while another range changes;
- selected range changed after the read;
- nonexistent or empty file behavior;
- existing callers that consume only `content` remain compatible.
