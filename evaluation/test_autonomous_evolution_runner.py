from pathlib import Path

from run_autonomous_evolution import (
    SYSTEM_PROMPT,
    _extract_execution,
    _suggest_capability_hook,
    compact_sensor_evidence_for_prompt,
)


def test_prompt_requires_post_lift_attachment_verification():
    assert "strict physical order" in SYSTEM_PROMPT
    assert "close, lift" in SYSTEM_PROMPT
    assert "Never call verify_attachment immediately after closing" in SYSTEM_PROMPT
    assert "baseline immutable across candidate retries" in SYSTEM_PROMPT


def test_prompt_preserves_full_instruction_entity_semantics():
    assert "Pass the live\ninstruction as a complete query" in SYSTEM_PROMPT
    assert "never replace it with a truncated token list" in SYSTEM_PROMPT
    assert "word-level\nqueries is a regression" in SYSTEM_PROMPT


def test_prompt_requires_preplacement_support_mask_verification():
    assert "retain its returned opaque mask_id" in SYSTEM_PROMPT
    assert "Pass that target_mask_id" in SYSTEM_PROMPT
    assert "containment and clearance" in SYSTEM_PROMPT
    assert "two consecutive fresh post-release" in SYSTEM_PROMPT
    assert "Do not return after the first verified frame" in SYSTEM_PROMPT


def test_prompt_forbids_uninspected_verified_prefix_regression():
    assert "inspect_controller_program on that best_prior is mandatory" in SYSTEM_PROMPT
    assert "preserve the bounded\nmulti-candidate loop" in SYSTEM_PROMPT


def test_prompt_requires_phase_labeled_motion_diagnostics():
    assert "Give every robot.act call a concise phase string" in SYSTEM_PROMPT
    assert "Do not leave failed actions or abort paths unlabeled" in SYSTEM_PROMPT


def test_prompt_routes_drawer_language_to_articulated_skill():
    assert "visual-articulated-drawer-open-and-retrieve Skill" in SYSTEM_PROMPT
    assert "capture_landmark_baseline" in SYSTEM_PROMPT
    assert "verify_landmark_displacement" in SYSTEM_PROMPT
    assert "least 4 cm of handle displacement" in SYSTEM_PROMPT
    assert "Fully reobserve and re-ground the source" in SYSTEM_PROMPT


def test_runner_injects_live_instruction_into_authoring_and_acquisition():
    source = Path(__file__).with_name("run_autonomous_evolution.py").read_text()
    assert 'task_instruction = libero_task_instruction("libero_spatial", args.task)' in source
    assert '"task_instruction": task_instruction' in source
    assert source.count("The live public task instruction is:") >= 2


def test_program_execution_counter_increments_once_per_round():
    source = Path(__file__).with_name("run_autonomous_evolution.py").read_text()
    assert source.count("program_executions += 1") == 1


def test_authoring_failure_is_preserved_without_becoming_robot_evidence():
    source = Path(__file__).with_name("run_autonomous_evolution.py").read_text()
    assert "thea_trace_authoring_failure_" in source
    assert '"consumed_as_experiment_round": False' in source
    assert "authoring produced no executed controller evidence" in source


def test_runner_preserves_partial_execution_directories_on_resume():
    source = Path(__file__).with_name("run_autonomous_evolution.py").read_text()
    assert 'while (round_dir / f"program_execution_{execution_index:03d}").exists()' in source
    assert 'output=round_dir / f"program_execution_{execution_index:03d}"' in source


def test_model_client_retries_transient_authoring_connection_failures():
    source = Path(__file__).with_name("run_autonomous_evolution.py").read_text()
    assert "max_retries=5" in source
    assert "timeout=300.0" in source
    assert 'reasoning_effort="low"' in source
    assert "max_tokens=5000" in source


def test_extract_execution_ignores_rejected_retry():
    events = [
        {"type": "tool_result", "name": "execute_controller_script", "result": {
            "success": True, "controller_id": "controller:v001",
            "sensor_evidence": {"sensor_only_conclusion": "transport_not_verified"},
        }},
        {"type": "tool_result", "name": "execute_controller_script", "result": {
            "success": False, "kind": "controller_execution_rejected",
            "reason": "round budget exhausted",
        }},
    ]
    controller, evidence = _extract_execution(events)
    assert controller == "controller:v001"
    assert evidence["sensor_only_conclusion"] == "transport_not_verified"


def test_failed_ranking_redirects_acquisition_to_execution_control():
    failure = {
        "failure_class": "attachment_not_verified",
        "capability_outcomes": [{
            "hook": "grasp_retry_ranking",
            "tool_id": "ranker:v001",
            "outcome": "failure_persisted",
        }],
    }
    assert _suggest_capability_hook(failure) == "grasp_execution_profile"
    assert _suggest_capability_hook({
        "failure_class": "attachment_not_verified", "capability_outcomes": []
    }) == "grasp_retry_ranking"
    assert _suggest_capability_hook({
        "failure_class": "transport_not_verified",
        "capability_outcomes": [{"hook": "transport_profile", "outcome": "failure_persisted"}],
    }) is None
    assert _suggest_capability_hook({
        "failure_class": "contact_convergence_failed", "capability_outcomes": []
    }) == "grasp_execution_profile"


def test_prompt_compaction_keeps_diagnostics_and_representative_actions():
    evidence = {
        "sensor_only_conclusion": "attachment_not_verified",
        "control_diagnostics": {"targets_not_reached": 30},
        "rpc_methods": ["act"] * 30 + ["observe"] * 4,
        "action_outcomes": [
            {"command_index": index, "final_error_m": index / 100.0}
            for index in range(30)
        ],
    }
    compact = compact_sensor_evidence_for_prompt(evidence, max_action_samples=9)
    assert compact["action_outcomes_total"] == 30
    assert compact["action_outcomes_omitted"] == 21
    assert len(compact["action_outcome_samples"]) == 9
    assert compact["rpc_method_counts"] == {"act": 30, "observe": 4}
    assert "action_outcomes" not in compact
    assert "rpc_methods" not in compact
