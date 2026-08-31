# Change

Not implemented.

Candidate minimal change:

- wrap model-facing `read_file` to retain its exact returned range digest;
- wrap model-facing `replace_file_lines` to supply that digest automatically;
- remove manual SHA carriage from the model-facing schema/manual;
- reject an Agent replacement lacking a matching prior read;
- leave `PersistentWorkspace.replace_file_lines` and its explicit guard intact.

No automatic content retry or conflict resolution is allowed.
