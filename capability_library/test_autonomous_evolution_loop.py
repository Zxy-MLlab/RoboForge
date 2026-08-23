import json
from pathlib import Path

from autonomous_evolution_loop import AutonomousEvolutionLoop, EvolutionConfig, sensor_only


def test_sensor_only_removes_nested_evaluator_fields():
    value = sensor_only({"success": True, "nested": {"reward": 1, "rgb": "frame"}})
    assert value == {"nested": {"rgb": "frame"}}


def test_loop_persists_rounds_and_stops_on_sensor_success(tmp_path: Path):
    calls = []

    def author(round_id, prior):
        calls.append((round_id, prior.get("sensor_evidence", {})))
        return {
            "controller_id": f"controller:v{round_id:03d}",
            "sensor_evidence": {
                "sensor_only_conclusion": "sensor_verification_passed" if round_id == 2 else "placement_not_verified",
                "success": True,
            },
            "tool_events": [{"name": "search_public_embodied_resources", "reward": 1}],
        }

    path = tmp_path / "state.json"
    state = AutonomousEvolutionLoop(
        EvolutionConfig("libero_spatial:task_3", max_rounds=5),
        state_path=path,
        author_round=author,
    ).run()
    assert state["status"] == "sensor_success"
    assert state["last_round"] == 2
    assert len(state["rounds"]) == 2
    assert calls[1][1]["sensor_only_conclusion"] == "placement_not_verified"
    persisted = json.loads(path.read_text())
    assert '"success"' not in json.dumps(persisted).lower()
    assert '"reward"' not in json.dumps(persisted).lower()


def test_loop_exhaustion_is_resumable(tmp_path: Path):
    calls = []

    def author(round_id, prior):
        calls.append(round_id)
        return {"sensor_evidence": {"sensor_only_conclusion": "attachment_not_verified"}}

    path = tmp_path / "state.json"
    config = EvolutionConfig("libero_spatial:task_1", max_rounds=2)
    loop = AutonomousEvolutionLoop(config, state_path=path, author_round=author)
    assert loop.run()["status"] == "exhausted"
    assert calls == [1, 2]
    assert loop.run()["status"] == "exhausted"
    assert calls == [1, 2]


def test_exhausted_loop_can_resume_with_larger_round_budget(tmp_path: Path):
    calls = []

    def author(round_id, prior):
        calls.append(round_id)
        return {"sensor_evidence": {"sensor_only_conclusion": "attachment_not_verified"}}

    path = tmp_path / "state.json"
    first = AutonomousEvolutionLoop(
        EvolutionConfig("libero_spatial:task_1", max_rounds=1),
        state_path=path,
        author_round=author,
    ).run()
    assert first["status"] == "exhausted"
    second = AutonomousEvolutionLoop(
        EvolutionConfig("libero_spatial:task_1", max_rounds=3),
        state_path=path,
        author_round=author,
    ).run()
    assert second["status"] == "exhausted"
    assert calls == [1, 2, 3]


def test_repeated_failure_triggers_separate_acquisition_and_full_history(tmp_path: Path):
    author_contexts = []
    acquisition_requests = []

    def author(round_id, context):
        author_contexts.append((round_id, context))
        return {
            "controller_id": f"controller:v{round_id:03d}",
            "sensor_evidence": {
                "sensor_only_conclusion": "placement_not_verified",
                "placement_verified": False,
                "capability_hook_invocations": [{
                    "hook": "support_relation_profile",
                    "tool_id": "strict_support:v001",
                    "applied": True,
                }],
            },
        }

    def acquire(round_id, request):
        acquisition_requests.append((round_id, request))
        return {"acquisition_completed": True, "tested_tool": "strict_support:v001"}

    path = tmp_path / "state.json"
    state = AutonomousEvolutionLoop(
        EvolutionConfig(
            "libero_spatial:task_5",
            max_rounds=3,
            acquisition_after_same_failure=2,
        ),
        state_path=path,
        author_round=author,
        acquire_capabilities=acquire,
    ).run()
    assert [item[0] for item in acquisition_requests] == [3]
    assert acquisition_requests[0][1]["consecutive_failures"] == 2
    assert acquisition_requests[0][1]["capability_outcomes"][0]["outcome"] == "failure_persisted"
    third_context = author_contexts[2][1]
    assert len(third_context["failure_history"]) == 2
    assert third_context["latest_acquisition"]["failure_class"] == "placement_not_verified"
    assert len(state["acquisitions"]) == 1


def test_restart_context_retains_latest_completed_acquisition(tmp_path: Path):
    loop = AutonomousEvolutionLoop(
        EvolutionConfig("task"), state_path=tmp_path / "state.json",
        author_round=lambda round_id, context: {},
    )
    context = loop._round_context(
        [{"round": 1, "sensor_evidence": {"sensor_only_conclusion": "failed"}}],
        None,
        [{"before_round": 2, "result": {"tested_tool": "recovery:v001"}}],
    )
    assert context["latest_acquisition"]["result"]["tested_tool"] == "recovery:v001"


def test_capability_outcome_credits_stage_progress_not_final_task():
    evidence = {
        "attachment_verified": True,
        "transport_verified": False,
        "capability_hook_invocations": [{
            "hook": "grasp_retry_ranking",
            "tool_id": "ranker:v001",
            "applied": True,
        }],
    }
    outcomes = AutonomousEvolutionLoop._capability_outcomes(evidence)
    assert outcomes == [{
        "hook": "grasp_retry_ranking",
        "tool_id": "ranker:v001",
        "invocations": 1,
        "applied_invocations": 1,
        "outcome": "stage_progressed",
    }]


def test_generic_articulation_capability_is_credited_from_visual_verification():
    evidence = {
        "verifications": [{"kind": "articulation", "verified": True}],
        "capability_hook_invocations": [{
            "hook": "generic_capability", "stage": "articulation",
            "tool_id": "pull_recovery:v001", "applied": True,
        }],
    }
    outcomes = AutonomousEvolutionLoop._capability_outcomes(evidence)
    assert outcomes == [{
        "hook": "generic_capability", "tool_id": "pull_recovery:v001",
        "stage": "articulation", "invocations": 1,
        "applied_invocations": 1, "outcome": "stage_progressed",
    }]


def test_generic_articulation_substage_is_credited_from_visual_verification():
    evidence = {
        "verifications": [{"kind": "articulation", "verified": True}],
        "capability_hook_invocations": [{
            "hook": "generic_capability", "stage": "articulation_recovery",
            "tool_id": "sensor_only_prismatic_articulation_recovery:v001",
            "applied": True,
        }],
    }
    outcomes = AutonomousEvolutionLoop._capability_outcomes(evidence)
    assert outcomes == [{
        "hook": "generic_capability",
        "tool_id": "sensor_only_prismatic_articulation_recovery:v001",
        "stage": "articulation_recovery", "invocations": 1,
        "applied_invocations": 1, "outcome": "stage_progressed",
    }]


def test_acquisition_failure_does_not_break_controller_iteration(tmp_path: Path):
    calls = []

    def author(round_id, context):
        calls.append((round_id, context.get("latest_acquisition")))
        return {"sensor_evidence": {"sensor_only_conclusion": "attachment_not_verified"}}

    def acquire(round_id, request):
        del round_id, request
        raise RuntimeError("research backend unavailable")

    state = AutonomousEvolutionLoop(
        EvolutionConfig("libero_spatial:task_3", max_rounds=2, acquisition_after_same_failure=1),
        state_path=tmp_path / "state.json",
        author_round=author,
        acquire_capabilities=acquire,
    ).run()
    assert len(calls) == 2
    assert state["acquisitions"][0]["result"]["acquisition_completed"] is False


def test_context_preserves_best_stage_when_latest_controller_regresses(tmp_path: Path):
    contexts = []

    def author(round_id, context):
        contexts.append(context)
        return {
            "controller_id": f"controller:v{round_id:03d}",
            "sensor_evidence": {
                "execution_completed": True,
                "attachment_verified": round_id == 1,
                "placement_verified": False,
                "sensor_only_conclusion": (
                    "placement_not_verified" if round_id == 1
                    else "attachment_not_verified"
                ),
            },
        }

    AutonomousEvolutionLoop(
        EvolutionConfig("task", max_rounds=3),
        state_path=tmp_path / "state.json", author_round=author,
    ).run()
    assert contexts[2]["prior_round"]["controller_id"] == "controller:v002"
    assert contexts[2]["best_prior"]["controller_id"] == "controller:v001"


def test_best_stage_preserves_visually_verified_articulation_prefix():
    opened = {
        "round": 2,
        "controller_id": "drawer_opened:v001",
        "sensor_evidence": {
            "execution_completed": True,
            "attachment_verified": False,
            "placement_verified": False,
            "verifications": [{
                "kind": "articulation", "verified": True,
                "horizontal_displacement_m": 0.048,
            }],
        },
    }
    regressed = {
        "round": 3,
        "controller_id": "drawer_regressed:v001",
        "sensor_evidence": {
            "execution_completed": True,
            "attachment_verified": False,
            "placement_verified": False,
            "verifications": [],
        },
    }
    best = AutonomousEvolutionLoop._best_prior([opened, regressed])
    assert best["controller_id"] == "drawer_opened:v001"


def test_acquisition_streak_prefers_mechanism_level_diagnosis():
    rounds = [{
        "sensor_evidence": {
            "sensor_only_conclusion": "attachment_not_verified",
            "diagnostic_failure_class": "drawer_open_not_verified",
        },
    }] * 2
    assert AutonomousEvolutionLoop._same_failure_streak(rounds) == (
        "drawer_open_not_verified", 2,
    )


def test_old_generic_evidence_is_upgraded_from_legal_phase_summary():
    evidence = {
        "sensor_only_conclusion": "development_run_completed_without_verification",
        "diagnostic_failure_class": "development_run_completed_without_verification",
        "phase_diagnostics": {"phases": {"contact": {
            "commands": 60, "reached": 3, "last_command_index": 80,
        }}},
    }
    assert AutonomousEvolutionLoop._effective_failure_class(evidence) == (
        "contact_convergence_failed"
    )
    assert AutonomousEvolutionLoop._same_failure_streak([
        {"sensor_evidence": evidence},
        {"sensor_evidence": {**evidence, "diagnostic_failure_class": "contact_convergence_failed"}},
    ]) == ("contact_convergence_failed", 2)


def test_authoring_infrastructure_failure_retries_same_round(tmp_path: Path):
    calls = []

    def author(round_id, context):
        del context
        calls.append(round_id)
        if len(calls) == 1:
            raise ConnectionError("temporary model transport error")
        return {
            "controller_id": "controller:v001",
            "sensor_evidence": {"sensor_only_conclusion": "attachment_not_verified"},
        }

    state = AutonomousEvolutionLoop(
        EvolutionConfig("task", max_rounds=1, max_authoring_attempts_per_round=2),
        state_path=tmp_path / "state.json", author_round=author,
    ).run()
    assert calls == [1, 1]
    assert len(state["rounds"]) == 1
    assert state["authoring_failures"] == [{
        "round": 1,
        "attempt": 1,
        "error": "ConnectionError: temporary model transport error",
        "consumed_as_experiment_round": False,
    }]


def test_force_acquisition_retries_after_harness_upgrade_without_deleting_history(tmp_path: Path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "protocol": "embodied-autonomous-evolution-v1",
        "task": "task", "status": "running",
        "evaluator_visible_to_agent": False,
        "rounds": [{
            "round": 1, "controller_id": "old:v001",
            "sensor_evidence": {
                "diagnostic_failure_class": "drawer_open_not_verified",
            },
        }],
        "acquisitions": [{
            "before_round": 2, "trigger_after_round": 1,
            "failure_class": "drawer_open_not_verified", "result": {},
        }],
        "authoring_failures": [],
    }))
    acquisitions = []

    def acquire(round_id, request):
        acquisitions.append((round_id, request))
        return {"implementation_attempted": True}

    def author(round_id, context):
        del round_id, context
        return {"sensor_evidence": {
            "diagnostic_failure_class": "drawer_open_not_verified",
        }}

    result = AutonomousEvolutionLoop(
        EvolutionConfig(
            "task", max_rounds=2, force_acquisition_next_round=True,
        ),
        state_path=state_path, author_round=author,
        acquire_capabilities=acquire,
    ).run()
    assert acquisitions[0][0] == 2
    assert acquisitions[0][1]["forced_after_harness_upgrade"] is True
    assert len(result["acquisitions"]) == 2
