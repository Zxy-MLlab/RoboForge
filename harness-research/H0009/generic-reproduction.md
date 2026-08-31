# Generic reproduction

The untouched H0008-derived baseline reproduced the issue with a FakeAdapter
whose authentic physical receipt disagrees with a Controller-local public
verifier.

Baseline expectations:

- physical AgentEvidence contains the local verifier result;
- physical AgentEvidence does not state authentic verification eligibility;
- `finish` knows the receipt is false, proving Harness already has the fact.

Candidate expectations:

- physical AgentEvidence and `inspect_execution` expose only
  `physical_verification.verified`;
- diagnostic evidence has no such field;
- no receipt metadata or private Adapter fields cross the model boundary;
- completion gate semantics are unchanged.

## Baseline result

At baseline `cd669787f30510914e947da167f2ccc3644abb08`, the seven-test
regression produced `5 failed, 2 passed`. All five failures were exact missing
`physical_verification` keys in immediate evidence, later inspection, or the
next-turn context. The two safety controls already passed: diagnostic evidence
did not fabricate the field, and `finish` remained bound to the private
authentic receipt.

## Candidate result

With the minimal candidate, the same test file produces `7 passed`. A wider
receipt/evidence/diagnostic closure set produces `73 passed`.
