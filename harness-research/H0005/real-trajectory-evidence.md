# Real trajectory evidence

Frozen campaign:
`/root/autodl-tmp/experiments/libero-real-20260830-h0005`

- Harness SHA: `4b98dfbd4a44336418dccb61fc67f14ac58bffda`.
- Real LIBERO tasks: 0, 1, and 2; development state 0.
- Model: `gpt-5.6-sol`; provider: Apex; reasoning effort: high.
- Agent calls: 219 + 205 + 148 = 572.
- Physical trials: 20 + 20 + 20 = 60.
- Diagnostic Tool calls: 17; 16 consumed diagnostic budget and 12 produced
  evidence.
- Task completion: 0/3.
- Guarded line replacements: 50 attempts, 22 successes, 28 failures.

Exact H0005 contingency:

| Supplied guard | Success | Failure |
|---|---:|---:|
| most recent same-range `read_file.range_sha256` | 22 | 0 |
| different/stale/guessed digest | 0 | 27 |
| malformed unrelated request | 0 | 1 |

All authoritative guards converted to edits. Every mismatched digest remained
fail closed. The model nevertheless spent 28 calls on failed replacements and
often switched to full-file writes, which succeeded 46/46.

Other generic interaction evidence:

- 49/601 Tool calls failed (8.15%).
- 157 model turns were explicit Decision Records.
- 10 physical-run requests were rejected for Decision linkage.
- 14 physical trials immediately reused the preceding Controller SHA.
- six physical trials contained syntax-invalid Controllers.
- no web acquisition occurred despite large remaining Agent-step budget.

The full trial-by-trial evidence and A/B/C/D/H analysis is in the campaign's
`forensic-analysis.md`.
