# H0006 candidate: Harness-threaded guarded line edits

## Status

Preregistered. Production code and tests are unchanged at baseline
`be5bffbef5dba0659f6fd1b83f851018ee79726e`.

## Real-trajectory problem

The frozen H0005 LIBERO campaign made 50 guarded line-replacement attempts.
All 22 calls that copied the most recent exact-range digest succeeded. All 27
calls carrying another digest failed closed; one malformed call failed for an
unrelated path error. The safety mechanism is correct, but the model still has
to transport opaque SHA metadata for a non-semantic editing operation.

## Hypothesis

For the Agent-facing workspace Tools, Harness can remember the most recent
successful `read_file` digest for each exact `(path, start_line, end_line)` and
thread it into the next matching `replace_file_lines` call. The model chooses
the range and replacement content; Harness handles concurrency metadata.

## Stronger-model counterfactual

An ideal model can copy H0005's digest, so this is not a hard execution block.
It is generic H interaction friction because digest carriage is mechanical,
causes redundant retries, and conveys no Controller/task decision.

## Safety invariant

Selected-range content changed after the read must still reject before write.
No matching read must fail closed at the Agent-facing boundary. Changes outside
the selected range may still merge, exactly as under H0005.
