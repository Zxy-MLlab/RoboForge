# Tests

Not written yet.

Required generic regressions:

- read -> unchanged same-range replace succeeds without model SHA;
- read -> changed same-range replace fails closed;
- read -> outside-range change -> same-range replace succeeds;
- replace without matching read fails closed;
- different range/path does not reuse a digest;
- explicit low-level H0005 guard remains supported;
- schema no longer asks the model to copy `expected_old_sha256`;
- cache loss/rebuilt AgentLoop fails closed.
