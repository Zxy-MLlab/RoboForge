# Real trajectory evidence

H0008 Task 3 produced repeated cases where Controller-local
`visual_attachment` and `visual_support_relation` were true while independent
public-sensor before/after outcome made the authentic physical receipt false.
The public AgentEvidence omitted the physical verification bit. The Agent
recorded working progress, made ten rejected `finish` calls, and immediately
reused the same Controller in 21/30 trials.

Task 4 independently showed local/independent disagreement. Task 7 supplied a
control: when local checks and authentic receipt aligned, the Agent completed
after five trials.

Source report:

`/root/autodl-tmp/experiments/libero-real-20260831-h0008/forensic-analysis.md`

The matched H0009 validation produced the following observable outcomes under
the same task/state/budget protocol: task 3 exhausted 30 trials after 359
calls; task 4 completed after 13 trials and 126 calls; task 7 exhausted 30
trials after 466 calls. This confirms that the projection is safe and truthful
but that its benefit is trajectory-dependent rather than a universal task
solver improvement.
