# H0003: Model-visible Controller entrypoint contract

## Problem

Physical and diagnostic execution both load `workspace/controller.py` and
require `def run(robot)`, but neither model-visible Tool description stated the
file or entrypoint contract.

## Hypothesis

Adding the exact shared source/entrypoint requirement to `run_controller` and
`run_diagnostic` descriptions will increase valid code-to-execution conversion
and useful diagnostic evidence without changing runtime behavior.

## Stronger-model counterfactual

An ideal embodied strategist still cannot derive a private Python loader
entrypoint from the Adapter action SDK. It must learn through a failed
execution. This is generic interface friction, not task reasoning.

## Generality gates

- Task deletion: every Controller task uses the same loader.
- Environment deletion: runtime entrypoint is Adapter-independent.
- Model deletion: function descriptions are model/provider-neutral.
- Strategy independence: no diagnostic content or action is prescribed.
- Real-robot realizability: only a software entrypoint is exposed.
- Minimality: two existing descriptions change; runtime is untouched.
