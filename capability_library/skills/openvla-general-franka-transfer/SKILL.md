---
name: openvla-general-franka-transfer
description: Use the public OpenVLA 7B base model with its documented non-LIBERO NYU Franka action normalization for task-disjoint robot manipulation transfer. Use when a development or evaluation task needs a frozen generalist VLA candidate and provenance has been checked.
---

# OpenVLA General Franka Transfer

Use only the public base checkpoint and the fixed
`nyu_franka_play_dataset_converted_externally_to_rlds` statistics. Do not use
LIBERO-specific checkpoints, LIBERO normalization keys, demonstrations, reward,
success, task IDs, or simulator state in the model call.

Feed the current legal RGB camera observation and natural-language instruction
to the adapter. It returns bounded 7-DoF OSC-style actions. Keep action chunks
short and rely on the Harness post-condition/evaluator outside model inference.

Record the model revision, stats key, action adapter, and every integration
failure. Development outcomes can improve generic adapters or search queries;
sealed outcomes cannot select prompts, scales, chunk lengths, or checkpoints.
