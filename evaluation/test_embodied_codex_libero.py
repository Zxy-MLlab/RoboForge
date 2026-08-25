import json
from pathlib import Path

import pytest

from evaluation.run_embodied_codex_libero import (
    _task_list,
    _campaign_exit_code,
    _development_must_halt,
    _prepare_campaign_root,
    _resolve_campaign_capability_library,
    _predeclared_partition,
    _resolve_packaging_skill,
    development_command,
    validation_command,
)


def test_resumed_campaign_recovers_legacy_task_capability_library(tmp_path):
    campaign=tmp_path/"campaign";campaign.mkdir()
    (campaign/"campaign.json").write_text(json.dumps({"protocol":"legacy"}))
    configured=tmp_path/"shared"/"tools"
    task=campaign/"task_02"/"development";task.mkdir(parents=True)
    (task/"harness_configuration.json").write_text(json.dumps({
        "capability_root":str(configured)}))
    assert _resolve_campaign_capability_library(
        output=campaign,requested=None)==configured.resolve()


def test_explicit_campaign_capability_library_overrides_legacy_inference(tmp_path):
    campaign=tmp_path/"campaign";campaign.mkdir()
    requested=tmp_path/"requested"/"tools"
    assert _resolve_campaign_capability_library(
        output=campaign,requested=requested)==requested.resolve()


def test_campaign_resolves_latest_audited_packaging_migration(tmp_path):
    family=tmp_path/"skill";controller_hash="a"*64
    for version,migration in [(1,None),(2,{"source_skill_id":"skill:v001",
            "controller_sha256_unchanged":True}),(3,{"source_skill_id":"skill:v001",
            "controller_sha256_unchanged":True})]:
        path=family/f"v{version:03d}";path.mkdir(parents=True)
        manifest={"skill_id":f"skill:v{version:03d}","version":version,
                  "controller_sha256":controller_hash}
        if migration:manifest["packaging_migration"]=migration
        (path/"manifest.json").write_text(json.dumps(manifest))
    path,manifest=_resolve_packaging_skill(family/"v001")
    assert path==family.resolve()/"v003"
    assert manifest["skill_id"]=="skill:v003"


def test_campaign_root_validation_uses_state_before_harness_creates_assets(tmp_path):
    fresh=tmp_path/"fresh"
    assert _prepare_campaign_root(fresh) is False
    (fresh/"assets"/"tools").mkdir(parents=True)
    occupied=tmp_path/"occupied";occupied.mkdir();(occupied/"user.txt").write_text("keep")
    assert _prepare_campaign_root(occupied) is True


def test_campaign_exit_code_distinguishes_frontier_from_infrastructure():
    frontier={"task_results":[{"task":0,"development_returncode":2,
        "status":"evolving"}]}
    assert _campaign_exit_code(frontier)==2
    assert frontier["capability_incomplete_tasks"]==[0]
    broken={"task_results":[{"task":0,"development_returncode":7,
        "status":"process_failed"}]}
    assert _campaign_exit_code(broken)==1
    assert broken["infrastructure_failures"][0]["phase"]=="development"


def test_campaign_halts_instead_of_skipping_task_on_development_process_failure():
    assert _development_must_halt(0) is False
    assert _development_must_halt(2) is False
    assert _development_must_halt(1) is True
    assert _development_must_halt(7) is True


def test_task_range_parser_is_deterministic():
    assert _task_list("3,5-7,5") == [3, 5, 6, 7]
    with pytest.raises(Exception):
        _task_list("10")


def test_development_and_sealed_state_partitions_are_disjoint_and_deterministic():
    first=_predeclared_partition(task=4,development_state=0,development_count=3,
        sealed_count=3,state_count=50,seed=2909)
    second=_predeclared_partition(task=4,development_state=0,development_count=3,
        sealed_count=3,state_count=50,seed=2909)
    assert first==second and first["development"][0]==0
    assert len(first["development"])==3 and len(first["sealed"])==3
    assert set(first["development"]).isdisjoint(first["sealed"])


def test_canonical_command_exposes_only_complete_program_controller():
    command = development_command(
        task=4,states=[23,7,9],max_iterations=8,output=Path("run"),
        capabilities=Path("tools"),model="gpt-5.6-sol",
        reasoning_effort="high",device="cuda",python="python",
        groundingdino_checkpoint="groundingdino.pth",base_url="https://api.example/v1",
    )
    assert "embodied_codex.examples.run_libero" in command
    assert "run_autonomous_evolution.py" not in command
    assert "--capability-library" in command
    assert command[command.index("--verifier-reasoning-effort")+1]=="low"
    assert command[command.index("--states")+1:command.index("--max-iterations")]==["23","7","9"]


def test_development_command_can_decouple_verifier_reasoning_effort():
    command = development_command(
        task=2,states=[0,4,49],max_iterations=20,output=Path("run"),
        capabilities=Path("tools"),model="gpt-5.6-sol",
        reasoning_effort="high",verifier_reasoning_effort="low",
        device="cuda",python="python",
        groundingdino_checkpoint="groundingdino.pth",
        base_url="https://api.example/v1")
    assert command[command.index("--reasoning-effort")+1]=="high"
    assert command[command.index("--verifier-reasoning-effort")+1]=="low"


def test_validation_command_requires_three_state_runner_inputs():
    command = validation_command(
        skill_dir=Path("skills/learned_transfer/v001"),task=5,states=[1,2,3],
        output=Path("validation"),model="gpt-5.6-sol",
        reasoning_effort="high",device="cuda",python="python",
        groundingdino_checkpoint="groundingdino.pth",base_url="https://api.example/v1",
    )
    assert "embodied_codex.examples.evaluate_libero_skill_sealed" in command
    assert command[command.index("--states")+1:command.index("--model")]==["1","2","3"]
