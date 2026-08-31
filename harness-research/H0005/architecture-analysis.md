# Architecture analysis

H0005 moves digest generation into Harness while leaving the semantic edit
decision with the model. It does not remove optimistic concurrency or infer
what content should be written.

The real campaign distinguishes two claims:

1. **Correctness/truthfulness:** solved. The model can obtain the exact digest,
   and 22/22 correctly threaded replacements succeeded.
2. **Low-friction interaction:** incomplete. The model still has to preserve
   range identity and copy an opaque SHA. Twenty-seven wrong-digest calls were
   rejected despite an otherwise recognizable edit intent.

Stronger-model counterfactual: an ideal model can use H0005 correctly, so this
is not a hard execution block. However, SHA carriage is mechanical rather than
semantic and creates redundant read/retry turns. A future hypothesis may test
server-side threading of the latest same-range read digest while applying the
same fail-closed comparison. That is not implemented here.

Generality gates pass for H0005: the mechanism remains relevant without
LIBERO, any particular Adapter, model, strategy, or simulator truth.
