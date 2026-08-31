# H0007 candidate: atomic consequential intent

## Status

Preregistered. Production code and tests remain at retained H0005 behavior,
with H0006 reverted, at `368d704f52ebb9ee685f215494a25f4784da0e0f`.

## Real-trajectory problem

The frozen three-task H0005 LIBERO campaign used 157 model responses solely to
call `record_decision`: 57, 48, and 52 per task. Every Decision response was a
standalone Tool call; none shared a response with the consequential operation.
Ten `run_controller` requests failed because the pending Decision protocol was
not satisfied. In nine cases the next successful physical attempt required no
new workspace operation between the rejection and a replacement
`record_decision -> run_controller` sequence. This is observable protocol
friction, not hidden reasoning evidence.

## Hypothesis

A consequential Tool request can carry the model's explicit structured
Decision context as part of that same request. Harness can durably create and
link the Decision Record immediately before invoking the requested operation.
The model still chooses and states the goal, evidence, hypothesis, decision,
expected effect, and uncertainty; Harness removes only the separate pending
state transition and round-trip.

## Stronger-model counterfactual

Even an ideal model with a correct task strategy must currently spend a
separate response establishing Harness pending state before it can express a
consequential action. Combining two explicit intents atomically removes
mechanical protocol work without choosing or fabricating the decision.

## Safety invariant

No consequential operation may execute unless its model-authored Decision
context validates and is durably recorded. The operation and evidence must be
linked to that exact Decision. Missing, malformed, or uncommittable context
must fail before mutation. READ_ONLY and VALIDATION Tools must remain free of
Decision requirements.
