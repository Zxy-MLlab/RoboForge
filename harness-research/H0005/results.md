# Results

Investigation only.

- Repeated field observations: 4/4 valid H0004 runs had one false guarded-edit
  rejection.
- Snapshot validation: 4/4 supplied digests differed from the authoritative old
  range despite no intervening mutation.
- Generic temporary-workspace reproduction: confirmed public-content digest
  mismatch and unguarded success.
- Static caller audit: no equivalent public range-digest mechanism exists and
  no repository caller requires a closed `read_file` result shape.
- Boundary audit: range-external changes remain mergeable; range-internal
  changes remain fail closed; newline normalization must be shared.
- Estimated context cost: approximately 81 serialized characters per read,
  648 characters across the eight reads in the four valid H0004 runs.
- Experimental design: deterministic safety regression followed by an
  unprompted autonomous FakeAdapter comparison; KEEP/REVERT gates are
  preregistered in `experiment-protocol.json`.
- Alternative analysis: removing/ignoring the guard, whole-file locking,
  displayed-content hashing, and automatic retry were rejected before coding.
- Production change: none.
- New regression: none.
- New embodied experiment: none.

KEEP/REVERT is not yet applicable. H0005 remains a candidate pending explicit
authorization to begin the test-first implementation/controlled-comparison
cycle.

## Stage 1 and Stage 2 update

Stage 1 is complete: the baseline regression failed before implementation and
the candidate's seven focused range-digest regressions pass. Full repository
validation after the implementation is `258 passed`, compileall passed, and
`git diff --check` passed.

Stage 2 ran the preregistered eight-run FakeAdapter autonomous A/B. Candidate
performed two guarded replacements successfully, each using a digest returned
by `read_file`; baseline attempted zero guarded replacements. The raw metrics
are in `/root/autodl-tmp/harness-validation-H0005/metrics.json` and the
comparison is in `/root/autodl-tmp/harness-validation-H0005/comparison.md`.

Stage 2 is **inconclusive**, not a positive causal result: baseline has no
success-rate denominator, candidate had two unrelated Tool errors, and overall
protocol-error/model-call improvements were not observed. H0005 is therefore
not yet KEEP-qualified under the preregistered autonomous gate. No LIBERO run
was performed.
