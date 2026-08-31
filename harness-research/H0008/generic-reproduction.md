# Generic reproduction

The completed forensic analysis selected one candidate for a separate H0009
generic reproduction: public physical-verification eligibility in
AgentEvidence. H0008 itself remains a no-change discovery iteration.

The required FakeAdapter reproduction must create a physical trial where a
Controller-local public verifier reports true but the Adapter's authentic
physical receipt is false. Baseline AgentEvidence must demonstrate that the
model cannot distinguish that state from completion-eligible evidence without
calling `finish`. The candidate may expose only a public boolean; it must not
expose receipt metadata or hidden evaluator state.
