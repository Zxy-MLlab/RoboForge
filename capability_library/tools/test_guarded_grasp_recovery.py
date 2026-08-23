import pytest

from guarded_grasp_recovery import guarded_pick_place_phases


def test_schedule_has_guarded_reobserve_and_recovery_phases():
    phases = guarded_pick_place_phases()
    names = [phase.name for phase in phases]
    assert names[:3] == ["pregrasp_reobserve", "guarded_approach", "close_and_settle"]
    assert "vertical_lift_verify" in names
    assert all(phase.reobserve_before and phase.stop_on_contact_anomaly for phase in phases)


def test_invalid_limits_rejected():
    with pytest.raises(ValueError):
        guarded_pick_place_phases(approach_steps=0)
