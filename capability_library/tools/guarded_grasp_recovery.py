"""Generic guarded grasp/recovery phase generator.

The planner emits sensor-driven phases only; a caller must provide RGB-D,
calibration and proprioception to implement each phase. It does not inspect
task IDs, rewards, success predicates, demonstrations, or simulator state.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardedPhase:
    name: str
    purpose: str
    max_steps: int
    gripper: float
    reobserve_before: bool
    stop_on_contact_anomaly: bool


def guarded_pick_place_phases(*, approach_steps: int = 80, lift_steps: int = 45, place_steps: int = 70) -> tuple[GuardedPhase, ...]:
    """Return a reusable closed-loop phase schedule."""
    if min(approach_steps, lift_steps, place_steps) <= 0:
        raise ValueError("phase step limits must be positive")
    return (
        GuardedPhase("pregrasp_reobserve", "relocalize object and clearance", 12, -1.0, True, True),
        GuardedPhase("guarded_approach", "slow approach along sensor-derived target direction", approach_steps, -1.0, True, True),
        GuardedPhase("close_and_settle", "close gripper and wait for proprioceptive plateau", 90, 1.0, True, True),
        GuardedPhase("vertical_lift_verify", "lift and verify attachment from legal observations", lift_steps, 1.0, True, True),
        GuardedPhase("relocalize_transport", "reobserve object and target before transport", 12, 1.0, True, True),
        GuardedPhase("guarded_place", "slow target approach with latest RGB-D geometry", place_steps, 1.0, True, True),
        GuardedPhase("release_observe", "release and observe placement stability", 35, -1.0, True, True),
    )


__all__ = ["GuardedPhase", "guarded_pick_place_phases"]
