# Real trajectory evidence

Source:
`/root/autodl-tmp/experiments/libero-real-20260830-h0005/`

Frozen retained Harness under observation:
`be5bffbef5dba0659f6fd1b83f851018ee79726e`

## Counts

| Metric | Task 0 | Task 1 | Task 2 | Total |
|---|---:|---:|---:|---:|
| Agent/model responses | 219 | 205 | 148 | 572 |
| Standalone `record_decision` responses | 57 | 48 | 52 | 157 |
| Decision responses containing another Tool | 0 | 0 | 0 | 0 |
| `run_controller` rejected for Decision state | 4 | 2 | 4 | 10 |

Decision-only responses were 27.4% of all model responses. The EventStore
contains 157 corresponding model-authored Decision Records; all 157 Tool calls
succeeded, so the cost is not caused by provider or schema failure.

For nine of ten rejected physical runs, the trace later shows a successful
`record_decision -> run_controller` recovery without an intervening workspace
mutation. The remaining rejection was followed by additional evidence
inspection and then the same recovery. The physical operation was blocked by
the explicit prerequisite transition.

## Scope limit

This evidence does not show that Decision provenance is unnecessary. It shows
that a separate pending-state Tool round-trip is unnecessary. H0007 must
preserve the complete explicit Decision payload and durable operation/evidence
links.
