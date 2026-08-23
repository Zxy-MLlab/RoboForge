# Harness capability-acquisition experiment

## Object under test

Test an Embodied Harness that starts with an LLM, legal sensors, a simulator
adapter, and an internet-capable software workspace. It must autonomously
acquire and compose capabilities to solve robot tasks. This is neither a
code-only baseline nor a leaderboard comparison of fixed pretrained policies.

Allow any public algorithm, model, policy, paper, repository, Tool, or Skill.
Learned perception, VLMs, grasp policies, motion planners, and robot foundation
models are explicitly allowed.

## Hard integrity boundary

Do not use learned parameters, adapters, normalization/retrieval statistics, or
demonstrations trained on an evaluated task. Do not expose reward, success,
goal predicates, privileged poses, simulator internals, or evaluation episode
identifiers to action selection. Do not manually supply a task solution after
an autonomous run starts.

Training on other tasks is permitted and disclosed. Report models with no
training in the benchmark family separately from same-family task-disjoint
transfer.

## Experimental loop

1. Run the initially deployed Harness on non-claimable development tasks.
2. Let it analyze failures and search public resources without a curated answer.
3. Require provenance checks before a learned asset enters the primary track.
4. Let it install, adapt interfaces, register Tools/Skills, and retry.
5. Retain failed acquisitions; do not report only successful assets.
6. Freeze the complete candidate before evaluating held-out tasks.
7. Run the sealed surface without using its outcomes for iteration.

## Measurements

- Initial/final task success and absolute acquisition gain.
- Frozen ablations attributing success to acquired assets.
- Searches, assets considered, integration attempts, and failed acquisitions.
- Capability reuse across task definitions and suites.
- Wall time, model/API calls, GPU hours, and storage.
- Human task-solving interventions after run start; target zero.
- Per-episode unresolved failures with a consistent bottleneck taxonomy.

The capability ceiling is the best sealed held-out score from a frozen,
provenance-compliant Harness, accompanied by its acquisition trace and all
failed tasks. Development performance measures learning dynamics, not the
final capability claim.
