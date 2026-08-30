# Tests

Regression:

`test_decision_schema_declares_opaque_evidence_reference_formats`

The baseline failed with missing `pattern`; the candidate exposes the exact
whitelist. Existing direct-handler rejection of host paths remains covered by
`test_decision_record_rejects_non_routing_evidence_reference`.

Primary metric: successful Decision Records / attempted Decision Records.
