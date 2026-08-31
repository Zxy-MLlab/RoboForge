# H0009 public physical-verification status

## Status

Preregistered. No production change has been implemented.

## Hypothesis

A physical trial's public AgentEvidence should state whether the authentic
Adapter sensor receipt is verified. Keeping that boolean entirely in Harness
metadata forces the model to infer completion eligibility from potentially
conflicting Controller-local checks or discover it by attempting `finish`.

The candidate is generic H interaction friction, not a task-solving feature.
It does not select actions, capabilities, targets, retries, or strategies.

## Invariant

For physical evidence only, project exactly one already-computed factual
status:

`physical_verification: {verified: bool}`

Receipt identity, Controller binding, environment identity, resume tokens,
hidden evaluator, reward, simulator truth, and sealed evaluation remain
private. Diagnostic evidence never receives physical completion status.

## Stronger-model counterfactual

Even an ideal model with a correct Controller can receive a local public check
that disagrees with the Adapter's authentic physical receipt. It should not
need to learn an opaque completion gate or spend an extra failed `finish` call
to know whether the just-returned evidence is completion-eligible.
