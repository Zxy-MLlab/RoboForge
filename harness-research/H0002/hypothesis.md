# H0002: Model-visible consequential-tool prerequisites

## Problem

ToolRegistry stores each Tool's consequence and AgentLoop enforces a pending
Decision Record, but the model-visible Tool schema exposes neither fact.

## Hypothesis

Appending the authoritative consequence and `record_decision` prerequisite to
consequential Tool descriptions will reduce avoidable protocol rejections
without weakening the Decision protocol or prescribing task behavior.

## Stronger-model counterfactual

An ideal task-solving model still cannot derive an undisclosed Harness
precondition from the Tool schema. It can learn only after a rejected call.
Making an existing mechanical contract visible removes that unnecessary round
trip while leaving all task decisions with the model.

## Generality gates

- Task deletion: mutation preconditions apply to every task.
- Environment deletion: ToolRegistry is Adapter-independent.
- Model deletion: function descriptions are provider/model-agnostic.
- Strategy independence: no action, object, Tool preference, or retry is added.
- Real-robot realizability: the change exposes Harness policy only.
- Minimality: descriptions are derived from existing metadata; no new API.
