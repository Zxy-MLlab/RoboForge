# Evidence

P30 trace:

- Run: `/root/autodl-tmp/experiments/libero-batch-20260830-p30-trace/dev-task1-state0`
- Failed references: `execution-000003`, `execution-000011`,
  `execution-000016`, `execution-000019`, and `execution-000022`.
- Successful control reference in the same turn: `execution-000001`.
- Exact error: `ProtocolError: checkpoint evidence artifact is missing or corrupt`.

All referenced execution JSON files existed and matched their stored SHA256.
The first preceding `decision_link` reused the evidence `artifact_uri` but did
not contain `artifact_sha256`. The unfiltered scan decoded that link before it
reached a later target execution.

Generic reproduction:

1. Commit a first physical execution and its Decision link.
2. Commit a second physical execution.
3. Inspect `evidence://execution-000002`.
4. Frozen baseline raises the same ProtocolError on the first Decision link.
