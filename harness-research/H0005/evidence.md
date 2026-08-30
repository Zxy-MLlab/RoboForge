# Evidence

## Controlled H0004 trajectories

All four valid baseline/candidate runs produced one
`WorkspaceError: file changed` from `replace_file_lines`. Snapshot
reconstruction proves no file mutation occurred between the relevant write/read
and edit; every supplied digest was simply different from the exact internal
range digest.

| Run | Step | Range | Supplied digest | Actual range digest |
|---|---:|---:|---|---|
| `baseline-task1-v2` | 17 | 30-30 | `sha256:placeholder` | `6baa37647fa7...` |
| `candidate-task1` | 10 | 3-3 | `e1ecb1e13e15...` | `2f20ba7a47e5...` |
| `baseline-task2-v2` | 11 | 13-17 | `e1b9daebd7c7...` | `de4dbbc4462e...` |
| `candidate-task2` | 9 | 89-95 | `8e2a9a8f94cc...` | `82c496a9cb4b...` |

The same extended P30 trajectory used unguarded line replacement successfully
multiple times. No observed model call supplied a verified correct range hash.

## Current mechanism

- `read_file` decodes text, calls `splitlines()`, rejoins selected lines with
  `"\n"`, and returns no digest.
- `replace_file_lines` calls `splitlines(keepends=True)` and hashes the exact old
  range before comparing `expected_old_sha256`.
- The Tool schema defines `expected_old_sha256` as an undescribed optional
  string.

The public content and internally hashed content can therefore differ even
when the file is unchanged.
