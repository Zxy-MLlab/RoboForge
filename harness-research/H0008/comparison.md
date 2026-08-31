# H0008 comparison

H0008 is a frozen discovery campaign, not a matched A/B. Comparisons with
H0005 are descriptive only because task sets and physical budgets differ.

| Metric | H0005 | H0008 |
|---|---:|---:|
| Tasks | 0, 1, 2 | 3, 4, 7 |
| Physical trials/task | 20 | 30 |
| Total model calls | 572 | 675 |
| Total physical trials | 60 | 65 |
| Completed tasks | 0/3 | 1/3 |
| Tool-call error rate | 8.15% | 8.77% |
| Guarded-edit conversion | 22/50 (44.0%) | 32/63 (50.8%) |
| Correct latest-range guard conversion | 22/22 | 32/32 |

Task 7 success demonstrates capability, but not improved Harness friction: the
Harness is the same retained H0005 surface. H0008 independently reproduces
manual edit-token friction and reveals a separate ambiguity between
Controller-local verifier facts and the hidden physical-completion eligibility
bit.
