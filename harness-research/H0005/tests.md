# Tests

Implemented and passing.

Primary test-first regression:

`test_read_range_digest_can_guard_atomic_line_replacement`

The behavior-level invariant should prove that a digest returned by a read of
an unchanged line range succeeds, while any intervening change or mismatched
range still fails closed. Schema coverage must prove the relationship is
model-visible rather than merely adding an internal field.

Covered boundary behavior:

- LF range with a final newline;
- last line without a final newline;
- CRLF input using the replacement path's authoritative normalization;
- selected range unchanged while another range changes;
- selected range changed after the read;
- nonexistent or empty file behavior;
- existing callers that consume only `content` remain compatible.

Validation at the implementation SHA:

- focused H0005 tests: 7 passed;
- full suite: 258 passed;
- `python -m compileall -q embodied_codex evaluation`: passed;
- `git diff --check`: passed.

The regression was confirmed failing on the baseline before production code
changed. Changed selected ranges still reject; unchanged same-range reads guard
replacement successfully.
