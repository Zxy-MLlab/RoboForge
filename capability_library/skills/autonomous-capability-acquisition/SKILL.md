---
name: autonomous-capability-acquisition
description: Search public internet resources, audit model and algorithm provenance, register reusable embodied Tools/Skills, and record acquisition attempts without leaking the evaluated robot task. Use when a Harness analyzes a failure and needs to autonomously find, integrate, or preserve a new perception, grasp, planning, policy, or procedural capability.
---

# Autonomous Capability Acquisition

Use the public-resource market tools before inventing a new learned component.
Search broadly for the observed failure mode, inspect the original repository
and model card, and record exact revisions and licenses. Treat search results as
leads, not evidence of eligibility.

Do not confuse provenance research with capability acquisition. A discovered
paper or repository is only a lead. Acquisition requires an implementation or
installation attempt plus an executable test. Use the evaluator-isolated
engineering workspace to write multi-file code, clone public repositories,
install dependencies into the workspace, inspect sensor-only failure artifacts,
and run unit or smoke tests. The workspace cannot see evaluator internals or
secrets.

The predefined runtime hooks are conveniences, not a closed capability list.
If no hook matches a deterministic remedy, create a generic JSON-in/JSON-out
Tool with explicit object `input_schema` and `output_schema`, test it with
`test_capability_tool`, and preserve its immutable ID for the next controller.
Suitable generic Tools include articulation recovery, waypoint generation,
collision-aware path post-processing, perception fusion, and diagnostic
algorithms. Do not end with “missing hook” before attempting this path.

For every candidate:

1. Record source URL, revision, artifact hash, inputs/outputs, license, training
   datasets, preprocessing statistics, and claimed benchmarks.
2. Run the provenance gate with the evaluated task identifiers. Reject any
   parameter, adapter, normalization statistic, or retrieval index trained on
   those tasks; reject unknown provenance until resolved.
3. Prefer a public implementation with a bounded adapter and an offline smoke
   test. Never expose reward, success, simulator state, or an evaluation episode
   identifier to the candidate.
4. Register a small Tool or Skill with a stable name, declared inputs, post-
   conditions, and an audit entry. Keep the original source and local hash.
5. Invoke the development integration hook and record its success or failure;
   then invoke the development retry hook only when at least one accepted asset
   integrates successfully. These hooks must not receive evaluator-only data.
6. Retry only on the non-claimable development surface. Freeze the complete
   candidate before a sealed run; sealed outcomes cannot select assets.

The phase may finish without a usable capability only after it has saved a
concrete failed engineering, installation, contract, unit-test, or smoke-test
result. Searching and registering a discovery-only lead is not a sufficient
completion condition.

Preserve unsuccessful searches, rejected assets, integration errors, and failed
reuse attempts. The capability library must show what the agent tried, not only
what happened to work. See `protocol/research_question.md` and
`protocol/reporting_tracks.md` for the experiment contract.
