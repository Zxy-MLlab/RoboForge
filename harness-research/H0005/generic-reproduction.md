# Generic reproduction

The regression is Adapter- and task-independent:

1. create a workspace text file;
2. call `read_file` for an explicit line range;
3. immediately call `replace_file_lines` for the same range using the returned
   `range_sha256`;
4. require the replacement to succeed;
5. mutate the selected range after the read and require the same guarded
   replacement to fail closed.

The baseline failed because `read_file` did not expose the digest calculated by
the replacement path. H0005 passes by returning the authoritative normalized
range digest. Tests also cover LF, CRLF, no final newline, external-range
changes, selected-range changes, empty/nonexistent files, and schema/manual
visibility.

This reproduction uses no LIBERO state, coordinate, object, manipulation
strategy, or physical behavior.
