# Contamination ledger

| Asset | Classification | Use in claims | Reason |
|---|---|---:|---|
| `/data/zxy/vla_agentic_harness_pi0_libero` | Test-adaptive / privileged collection | No | Uses privileged simulator state, targeted failed initial-state collection, and LIBERO fine-tuning. |
| `/data/zxy/models/lerobot/pi05_libero_finetuned` | LIBERO-trained frozen policy | Benchmark-exposed reference only | Training config names `HuggingFaceVLA/libero`. |
| Cached `openvla-7b-oft-finetuned-libero-*` | LIBERO-trained frozen policy | Benchmark-exposed reference only | Suite-specific LIBERO OFT checkpoints. |
| Cached `openvla/openvla-7b` | Unresolved general base policy | Not until provenance resolves | Must verify training datasets and obtain non-LIBERO action normalization. |

## Protocol supersession — 2026-08-20 (historical)

The user subsequently prohibited all trained low-level robot models. Every
π0.5 and OpenVLA run is therefore invalidated, including partially completed
frozen evaluations. These artifacts are retained only for auditability and
must never be included in capability claims.

## Protocol correction — 2026-08-20

The user clarified that learned models are allowed; only models trained on the
current evaluated task are forbidden. Protocol v2 therefore admits frozen
general VLM, SAM/DINO-style perception, generic learned grasp policies, and
generalist robot policies whose parameters and preprocessing statistics have
no exposure to the evaluated tasks. Same-family, task-disjoint transfer is
allowed but reported separately. The earlier blanket prohibition is retained
above only as historical audit text.

The π0.5 LIBERO-finetuned and OpenVLA-OFT LIBERO checkpoints remain excluded
because they are actually LIBERO-trained. The `pi05_libero_base` asset and
base OpenVLA are not automatically admitted: their action normalization and
training provenance must pass the new gate first.

This ledger is append-only during the study. Reclassification must retain the
old entry and add dated evidence.
# 2026-08-20 — Code-only protocol clarification

- User prohibited every trained low-level model; active work now also avoids
  learned perception models.
- π0.5 and OpenVLA-OFT runs are audit-only and excluded from every capability
  claim, comparison, checkpoint selection, and benchmark conclusion.
- Active controller inputs are restricted to language, RGB-D, documented
  camera calibration, and robot proprioception. Object-state observations,
  rewards, done/success signals, BDDL predicates, task IDs, and initial-state
  IDs are unavailable to action selection.
- Development task 0 state 0 has been observed and used for debugging; it is
  non-claimable and must not appear in the frozen evaluation manifest.
