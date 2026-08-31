# Tests

Pre-campaign gates:

- full retained-Harness pytest must pass;
- compileall must pass;
- diff check must pass;
- LIBERO/Apex/sandbox doctor must pass;
- selected task/state environments must initialize under the same Adapter.

No candidate regression was added because H0008 changed no production code.

Forensic validation completed:

- 1,689 EventStore records passed chained-digest verification;
- three checkpoint envelopes passed SHA verification;
- 73 evidence references, 65 Controller snapshots, and 2,096 artifact entries
  existed and matched their recorded SHA;
- all three model histories contained zero compacted turns;
- 32/32 exact latest-range guarded replacements succeeded and 31/31 other
  replacement attempts failed closed.

The extraction is reproducible with external `analyze_campaign.py`; its result
is `forensic-metrics.json`.
