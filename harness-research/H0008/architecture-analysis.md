# H0008 architecture analysis

## Stronger-model counterfactual

An ideal model that already knew a correct manipulation strategy could express
numeric/pose actions, invoke current native capabilities, inspect RGB-D and
rollout artifacts, and execute physical trials. Task 7 proves this path is not
generally blocked. Task 3/4 failure alone therefore does not justify a Harness
change.

Two generic interaction costs nevertheless recur:

1. Models must manually carry exact range digests. The stale-write guard is
   correct, but only 32/63 replacement intents converted. H0006 already tested
   one server-threading design and rejected it because autonomous A/B did not
   lower errors or calls. H0008 does not reopen that rejected design.
2. Episodic `run_controller` resets internally, while its model-visible Tool
   description does not say so and `reset_case` remains separately visible.
   Eighteen explicit resets were redundant. This is a real hidden-contract
   cost, but lower impact than the evidence contradiction and does not justify
   selecting a second simultaneous hypothesis.

## Selected candidate for generic reproduction

Select exactly one candidate:

> A physical trial's public AgentEvidence should state whether its current
> sensor evidence is eligible for physical completion, without exposing the
> receipt token, environment identity, hidden evaluator, reward, or sealed
> evaluation result.

Today, `verification_receipt` is intentionally Harness metadata. That is
correct for provenance and recovery. The problem is that the public projection
contains no corresponding boolean status. A Controller-local verifier may say
true while the Adapter's independent public-sensor gate says false. The Agent
can then reasonably interpret a generic `finish` rejection as stale routing
rather than a factual trial failure.

This candidate is generic because it concerns the meaning of physical evidence,
not LIBERO labels, coordinates, capabilities, or manipulation policy. The
model still chooses every Controller and physical strategy. Harness only
projects one already-computed factual status.

## Security boundary

A candidate must expose at most a task-independent public status such as
`physical_verification: {verified: bool}`. It must not expose receipt identity,
environment tokens, hidden simulator success, reward, sealed evaluation, or
private sensor report fields. Diagnostic evidence must not receive or
fabricate physical completion status.

The candidate is not accepted. It first requires a generic failing FakeAdapter
regression and an autonomous A/B KEEP gate. If it fails to reduce
misinterpretation/retry friction, Core remains unchanged.
