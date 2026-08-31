# H0009 results

## Generic gate

The focused regression passed 7/7 and the affected closure set passed 73/73.
The corrected autonomous FakeAdapter A/B passed all safety gates and showed a
clear reduction in contradictory-evidence friction: 55 to 24 model calls,
11 to 4 physical trials, and 1/4 to 4/4 valid completions in the disagreement
condition. The control condition remained 4/4 valid completions with 30 versus
29 model calls. No Tool/protocol error or public private-metadata leak was
observed.

## Matched real validation

The frozen candidate was run against the exact H0008 task order, state,
provider, model, capability snapshot, sandbox, reset protocol, and budgets.

| Task | H0008 | H0009 |
|---:|---|---|
| 3 | 228 steps / 30 trials / false | 359 steps / 30 trials / false |
| 4 | 373 steps / 30 trials / false | 126 steps / 13 trials / true |
| 7 | 74 steps / 5 trials / true | 466 steps / 30 trials / false |

H0009 aggregate: 951 model calls, 1,037 Tool calls, 101 failed Tool calls,
73 physical trials, 15 diagnostic attempts (11 committed), 23 unchanged
immediate Controller reruns, and 1/3 valid completions. H0008 aggregate:
675 model calls, 718 Tool calls, 63 failed Tool calls, 65 physical trials,
11 diagnostic attempts (8 committed), 31 unchanged reruns, and 1/3 valid
completions.

## Disposition

The candidate is retained as a narrow, truthful public-evidence improvement:
all 73 physical evidence objects expose the authenticated boolean, all 11
diagnostics omit it, and private receipt/evaluator metadata remains isolated.
The real campaign result is mixed, not a general LIBERO score or friction
improvement. No further H0009 Harness redesign or task-specific tuning is
authorized by this experiment.

Raw campaign data and forensic metrics are retained outside the repository at
`/root/autodl-tmp/experiments/libero-real-20260831-h0009/`.
