# Change

`record_decision.evidence_refs.items` now declares:

- pattern `^(evidence|artifact|run)://`;
- a description identifying these as opaque references returned by Harness
  tools.

The independent `_record_decision` validation remains unchanged and fail
closed. Empty evidence lists remain valid. No provenance requirement is
weakened.
