# LIBERO results

## Matched frozen validation

Campaign: `/root/autodl-tmp/experiments/libero-real-20260831-h0009`
Frozen production Harness SHA: `5ef4c242f4bc5c1867538d92bcae9f03a5e48664`
Capability snapshot tree SHA: `2f82569eb68dbaa17703b62c38496ea20d8f38950bdd8066605ac8851e9d2d0d`

| Task | Agent steps | Physical trials | Diagnostics committed | Completion |
|---:|---:|---:|---:|---|
| 3 | 359 | 30/30 | 3 | false |
| 4 | 126 | 13/30 | 4 | true |
| 7 | 466 | 30/30 | 4 | false |

Aggregate H0009: 951 model calls, 1,037 Tool calls, 101 failed Tool calls,
73 physical trials, 15 diagnostic attempts (11 committed), 23 immediate
unchanged-Controller reruns, and 1/3 valid completions. All three task results
and the campaign status were durably recorded; no task returned an
infrastructure failure.

H0008 matched aggregate was 675 model calls, 718 Tool calls, 63 failed Tool
calls, 65 physical trials, 11 diagnostic attempts (8 committed), 31 immediate
unchanged-Controller reruns, and 1/3 valid completions. H0009 therefore did not
improve aggregate friction across this small set: model calls, Tool failures,
physical trials, and unchanged reruns increased. Task 4 alone improved from
373 steps/30 trials/false to 126 steps/13 trials/true.

The H0009 field-integrity audit passed: all 73 physical evidence objects carry
`physical_verification: {verified: bool}` matching the private authentic receipt;
all 11 committed diagnostics omit it; no model-visible result contained
`verification_receipt`, `environment_identity`, `episode_id`, or `resume_token`.

Raw forensic metrics are retained at
`/root/autodl-tmp/experiments/libero-real-20260831-h0009/forensic-metrics.json`.
