# Static design audit

## Existing alternatives

- `list_files` can expose a whole-file digest, not the selected line-range
  digest required by `replace_file_lines`.
- `write_file` is atomic but has no optimistic concurrency guard.
- No other public Tool returns the authoritative selected-range digest.

Therefore the issue cannot be resolved truthfully by documenting an existing
equivalent path.

## Required semantics

The digest must cover exactly the old text that `replace_file_lines` will
compare for the returned `start_line` and `end_line`, using the same newline
normalization. It is a range guard, not a whole-file lock.

Boundary checks confirmed:

- LF, CRLF, and CR inputs require digest computation from the same normalized
  representation used by replacement.
- A change outside the selected range may still merge successfully.
- A change inside the selected range must remain fail closed.
- Empty/nonexistent files must not advertise a usable replacement-range digest.
- Existing CRLF-to-LF normalization during replacement is pre-existing behavior
  and is outside H0005.

## Compatibility

Adding a result field is backward compatible with current callers. Repository
consumers either forward the result or read `content`; none validates an exact
closed result shape. The existing optional request field can remain optional.

The Tool parameter description should mechanically identify the returned field
as its source. No Decision, consequence, sandbox, workspace mutation, or
atomic-write behavior needs to change.

## Security

The digest covers text already returned to the model. It reveals no additional
host path or private content. Mismatches still reject before mutation, and the
proposal does not introduce automatic retry or overwrite behavior.

## Context cost

A `range_sha256` field adds approximately 81 serialized characters per
`read_file` result. The four valid H0004 runs contained eight reads, for about
648 additional characters before provider tokenization. This is bounded and
directly actionable metadata rather than replayed history.

## Minimal implementation boundary

If implementation is authorized, compute the returned digest through the same
text/range normalization path used by replacement and expose the relationship
in the existing Tool descriptions/schema. Do not create a new edit API or
change the guard from range-level to file-level.
