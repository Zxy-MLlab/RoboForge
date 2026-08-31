# Change

Candidate `1e7b2cbf48e90d80632884c226c8d4d78e552de0` implemented the following
minimal experiment:

- wrap model-facing `read_file` to retain its exact returned range digest;
- wrap model-facing `replace_file_lines` to supply that digest automatically;
- remove manual SHA carriage from the model-facing schema/manual;
- reject an Agent replacement lacking a matching prior read;
- leave `PersistentWorkspace.replace_file_lines` and its explicit guard intact.

No automatic content retry or conflict resolution is allowed.

The candidate passed its generic safety regressions but was reverted after the
autonomous A/B. The exact-range cache did not cover the common broad-read then
narrow-edit sequence, so it replaced manual digest errors with exact-range-read
prerequisite errors rather than reducing interaction friction.
