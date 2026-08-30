# Generic reproduction

Using a temporary `PersistentWorkspace` with `controller.py` equal to
`alpha\nbeta\n`:

1. `read_file(..., 1, 1)` returns `content="alpha"` and no digest.
2. SHA256 of the public content is `8ed3f6ad685b...`.
3. The internal old-range digest is SHA256 of `alpha\n`,
   `b6a98d9ce9a2...`.
4. Passing the public-content digest to `expected_old_sha256` raises
   `WorkspaceError: file changed`.
5. Omitting the guard succeeds.

This reproduction uses no Adapter, task, sensor, or model behavior.
