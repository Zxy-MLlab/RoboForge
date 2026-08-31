# Generic reproduction

Planned failing baseline behavior:

1. Agent-facing `read_file(path, start, end)` returns a range.
2. An external/concurrent writer changes that selected range.
3. Agent-facing `replace_file_lines(path, start, end, new_content)` omits any
   model-supplied digest.
4. Baseline overwrites because the optional guard is absent.
5. H0006 must reject with `WorkspaceError: file changed` by threading the
   stored read digest internally.

Control cases:

- unchanged selected range succeeds;
- change outside selected range succeeds;
- no matching Agent read fails closed;
- explicit programmatic `PersistentWorkspace` callers retain H0005 behavior;
- process/cache loss fails closed rather than becoming unguarded.
