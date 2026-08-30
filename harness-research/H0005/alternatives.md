# Alternatives considered

## Remove `expected_old_sha256` from the model schema

Rejected. This would reduce false guarded attempts by removing the safer
operation rather than making it usable. Programmatic compatibility would also
diverge from the model-facing contract.

## Use the whole-file digest from `list_files`

Rejected. The current invariant is range-level optimistic concurrency. A
whole-file guard would reject safe non-overlapping changes and `list_files`
does not guarantee a populated digest for every unsnapshotted file.

## Hash the displayed `read_file.content`

Rejected. Displayed content omits original line terminators, while replacement
currently hashes normalized old lines with terminators. Naming that digest
authoritative would preserve the observed mismatch.

## Describe the algorithm but return no digest

Rejected as insufficient. The model would still need another Tool turn to
compute SHA256 and would need to reproduce private newline normalization. The
goal is truthful low-friction guarded editing, not a cryptography exercise.

## Ignore a mismatched digest or retry automatically

Rejected. Either behavior weakens fail-closed concurrency protection and can
overwrite a change the model did not inspect.

## Return an opaque file revision token

Rejected for this iteration. It introduces a new concurrency abstraction and
would require defining whole-file versus range semantics. The existing range
digest can satisfy the demonstrated invariant with a smaller change.

## Preserve and return exact line-ending bytes as `content`

Rejected. It changes the established human/model-facing text representation
and can add formatting friction. A separate authoritative digest is smaller and
backward compatible.

## Make the guard required

Rejected. Existing callers intentionally omit it, and newly created or
uninspected files may require full atomic writes. H0005 should make the safe
option usable without changing when it is mandatory.
