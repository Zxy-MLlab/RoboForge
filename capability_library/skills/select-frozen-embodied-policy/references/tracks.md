# Evaluation tracks

## Generalist task zero-shot

Allow frozen learned models and code modules whose full known provenance has no
exposure to the currently evaluated tasks. Examples include a generic VLM,
SAM/DINO, a grasp model trained on GraspNet or synthetic grasp data, and a
robot policy trained on other tasks. Record preprocessing and action
normalization data as training exposure too. Report benchmark-family-disjoint
and task-disjoint-transfer assets separately.

## Classical diagnostic

Use no learned model. Report this separately as an ablation; it is not the
generalist capability ceiling.

## Benchmark-exposed reference

Place any asset trained, fine-tuned, normalized, distilled, or selected on the
currently evaluated task here. It is non-primary even when frozen at inference.

## Unknown provenance

Reject from the primary track until public documentation or author-provided
metadata resolves the training datasets. Absence of evidence is not evidence
of disjoint training.

## Disallowed in every track

Disallow inference that consumes evaluator-only success, reward, goal
predicates, privileged poses, simulator internals, or evaluation episode IDs.
Disallow parameter training on benchmark development/evaluation trajectories,
manual episode-specific solutions, and selection from sealed outcomes.
