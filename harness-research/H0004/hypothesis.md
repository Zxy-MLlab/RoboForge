# H0004: Truthful Decision evidence-reference schema

## Problem

`record_decision` accepts only opaque `evidence://`, `artifact://`, and
`run://` references, but its model-visible schema declared arbitrary strings.

## Hypothesis

Projecting the existing prefix whitelist into JSON Schema will increase valid
Decision conversion and reduce retry turns without changing accepted inputs.

## Stronger-model counterfactual

An ideal task strategist cannot infer a private routing grammar from an
unconstrained string schema. It must either omit legitimate evidence or learn
through rejection. This is generic interface friction.

## Generality gates

- Task/environment/model deletion: Decision provenance is Kernel-wide.
- Strategy independence: no task conclusion or required evidence is added.
- Real-robot realizability: references route public Harness evidence only.
- Minimality: schema mirrors the existing runtime whitelist.
