# H0001: Authoritative execution-event filtering

## Problem

`inspect_execution(ref)` can fail on valid later evidence because it attempts to
decode non-evidence provenance events that happen to carry `artifact_uri`.

## Hypothesis

Restricting evidence lookup to authoritative `execution` events will preserve
strict digest validation while allowing later committed executions to be
inspected reliably.

## Stronger-model counterfactual

An ideal model that supplies the exact valid evidence reference is still
blocked by an earlier `decision_link` without an artifact digest. This is an A
correctness issue, not a model strategy or capability issue.

## Generality gates

- Task deletion: evidence routing is needed without the observed task.
- Environment deletion: every Adapter uses the same EventStore lookup.
- Model deletion: the failure is independent of model choice.
- Strategy independence: no action, object, or Tool preference is encoded.
- Real-robot realizability: only durable public evidence is read.
- Minimality: one event-kind guard is sufficient.
