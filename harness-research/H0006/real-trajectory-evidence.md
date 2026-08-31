# Real trajectory evidence

Source campaign:
`/root/autodl-tmp/experiments/libero-real-20260830-h0005`.

- 50 `replace_file_lines` attempts;
- 22 exact most-recent same-range guards, all successful;
- 27 different/stale/guessed guards, all rejected;
- one unrelated malformed request;
- 28 failed edit calls and additional read/retry turns;
- whole-file writes succeeded 46/46 and were frequently used as escape paths.

The issue is workspace-protocol friction and contains no embodied task
knowledge.
