from reanalyze_program_motion import reanalyze_rpc_motion


def test_reanalysis_uses_rpc_proprioception_and_flags_premature_transition():
    report = reanalyze_rpc_motion({
        "rpc_events": [
            {"id": 1, "method": "observe", "result": {"eef_xyz": [0.0, 0.0, 1.2]}},
            {
                "id": 2, "method": "act",
                "arguments": {"action": {
                    "target_eef_xyz": [0.2, 0.2, 1.0], "repeat": 15,
                    "gripper": -1, "orientation": "topdown",
                }},
                "result": {"eef_xyz": [0.05, 0.05, 1.15], "step": 15},
            },
        ]
    })
    assert report["control_diagnostics"]["commands"] == 1
    assert report["control_diagnostics"]["targets_not_reached"] == 1
    assert report["action_outcomes"][0]["reached_target"] is False
    assert report["evaluator_used"] is False
