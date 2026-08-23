"""Minimal GPT-planned, RGB-D-localized LIBERO trajectory collector.

GPT supplies a validated symbolic JSON phase plan. RGB-D and declared camera
calibration supply positions. A local proportional move_to controller emits
bounded OSC actions. No Qwen, demonstrations, object-state, reward, or success
signal is used while selecting actions.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

import h5py
import numpy as np
from rgbd_pick_place import (
    CircularCandidate, allowed_observation, backproject_rgbd,
    detect_circular_candidates, estimate_table_height,
    parse_pick_place_instruction, segment_workspace_regions, select_for_intent,
)


ALLOWED_TARGETS = {"source_above", "source_grasp", "source_lift", "target_above", "target_place", "home"}


def ask_gpt_plan(instruction: str, source: np.ndarray, target: np.ndarray, table_z: float, *, endpoint: str, model: str) -> dict:
    key = os.environ.get("APEXIN_API_KEY")
    if not key:
        raise RuntimeError("APEXIN_API_KEY is required")
    prompt = f"""You are a robot trajectory planner. Return JSON only.
Plan a robust pick-and-place using symbolic targets. Allowed target values:
source_above, source_grasp, source_lift, target_above, target_place, home.
Each phase must have: name, target, gripper (-1 open or 1 close), max_steps (5..120), gain (1..20), settle_steps (0..30).
Use pregrasp, slow grasp, close/settle, vertical lift, transport above target, slow place, open/settle, retreat/home.
Do not use task IDs, rewards, simulator state, demonstrations, or invent coordinates.
Instruction: {instruction}
RGB-D source robot-frame point: {source.tolist()}
RGB-D target robot-frame point: {target.tolist()}
RGB-D table height: {table_z:.6f}
Schema: {{"phases":[{{"name":"...","target":"source_above","gripper":-1,"max_steps":60,"gain":8,"settle_steps":0}}]}}
"""
    payload = {
        "model": model,
        "reasoning_effort": "medium",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1800,
    }
    request = Request(endpoint.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode())
    text = body["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return validate_plan(json.loads(text))


def validate_plan(plan: dict) -> dict:
    phases = plan.get("phases")
    if not isinstance(phases, list) or not 5 <= len(phases) <= 16:
        raise ValueError("plan must contain 5..16 phases")
    clean = []
    for item in phases:
        target = str(item.get("target"))
        if target not in ALLOWED_TARGETS:
            raise ValueError(f"forbidden target: {target}")
        gripper = float(item.get("gripper"))
        if gripper not in (-1.0, 1.0):
            raise ValueError("gripper must be -1 or 1")
        clean.append({
            "name": str(item.get("name", target))[:64],
            "target": target,
            "gripper": gripper,
            "max_steps": int(np.clip(int(item.get("max_steps", 50)), 5, 120)),
            "gain": float(np.clip(float(item.get("gain", 8)), 1, 20)),
            "settle_steps": int(np.clip(int(item.get("settle_steps", 0)), 0, 30)),
        })
    return {"phases": clean}


def locate(rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray, instruction: str):
    world = backproject_rgbd(depth, intrinsic, extrinsic)
    table_z = estimate_table_height(world)
    candidates = detect_circular_candidates(rgb, world)
    if len(candidates) < 3:
        regions, _ = segment_workspace_regions(rgb, world, table_height=table_z)
        candidates.extend(CircularCandidate(
            center_rc=r.centroid_rc, radius_px=float(np.sqrt(r.area_px / np.pi)), interior_rgb=r.mean_rgb,
            darkness=float(np.clip(1 - np.mean(r.mean_rgb) / 255, 0, 1)),
            achromaticity=float(np.clip(1 - (max(r.mean_rgb)-min(r.mean_rgb))/80, 0, 1)),
            center_world=r.median_world,
        ) for r in regions)
    source, target = select_for_intent(parse_pick_place_instruction(instruction), candidates, table_z)
    return np.asarray(source.center_world), np.asarray(target.center_world), table_z


def main() -> None:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://api.apexin.ai/v1")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--grasp-height-above-table", type=float, default=0.055)
    parser.add_argument("--place-height-above-table", type=float, default=0.055)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, camera_depths=True, ignore_done=True, horizon=1400)
    observations, actions, phases, trace = [], [], [], []
    try:
        raw = env.reset()
        raw = env.set_init_state(suite.get_task_init_states(args.task)[args.state])
        obs = allowed_observation(raw)
        rgb = np.ascontiguousarray(obs["agentview_image"][::-1])
        depth = get_real_depth_map(env.sim, np.ascontiguousarray(obs["agentview_depth"][::-1]))
        intrinsic = get_camera_intrinsic_matrix(env.sim, "agentview", 256, 256)
        extrinsic = get_camera_extrinsic_matrix(env.sim, "agentview")
        source, target, table_z = locate(rgb, depth, intrinsic, extrinsic, task.language)
        plan = ask_gpt_plan(task.language, source, target, table_z, endpoint=args.endpoint, model=args.model)
        (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")

        source_xy, target_xy = source[:2], target[:2]
        grasp_z = table_z + args.grasp_height_above_table
        target_z = table_z + args.place_height_above_table
        home = np.asarray(obs["robot0_eef_pos"], dtype=float)
        goals = {
            "source_above": np.r_[source_xy, table_z + 0.22], "source_grasp": np.r_[source_xy, grasp_z],
            "source_lift": np.r_[source_xy, table_z + 0.24], "target_above": np.r_[target_xy, table_z + 0.24],
            "target_place": np.r_[target_xy, target_z], "home": home,
        }
        step = 0
        for phase_index, phase in enumerate(plan["phases"]):
            goal = goals[phase["target"]]
            for local_step in range(phase["max_steps"]):
                delta = (goal - np.asarray(obs["robot0_eef_pos"])) * phase["gain"]
                action = np.zeros(7, dtype=np.float32)
                action[:3] = np.clip(delta, -0.35, 0.35)
                action[6] = phase["gripper"]
                observations.append({k: np.asarray(v).copy() for k, v in obs.items()})
                actions.append(action.copy()); phases.append(phase_index)
                raw, _, _, _ = env.step(action)
                obs = allowed_observation(raw)
                trace.append({"step": step, "phase": phase["name"], "target": phase["target"], "goal": goal.tolist(), "eef": obs["robot0_eef_pos"].tolist(), "action": action.tolist()})
                step += 1
                if np.linalg.norm(goal - obs["robot0_eef_pos"]) < 0.006 and local_step >= 3:
                    break
            for _ in range(phase["settle_steps"]):
                action = np.zeros(7, dtype=np.float32); action[6] = phase["gripper"]
                observations.append({k: np.asarray(v).copy() for k, v in obs.items()}); actions.append(action.copy()); phases.append(phase_index)
                raw, _, _, _ = env.step(action); obs = allowed_observation(raw); step += 1

        success = bool(env.check_success())
        with h5py.File(args.output / "trajectory.hdf5", "w") as h5:
            group = h5.create_group("data/demo_0")
            group.create_dataset("actions", data=np.asarray(actions), compression="gzip")
            group.create_dataset("phase_index", data=np.asarray(phases, dtype=np.int16))
            obs_group = group.create_group("obs")
            for key in observations[0]:
                obs_group.create_dataset(key, data=np.asarray([item[key] for item in observations]), compression="gzip")
            group.attrs["instruction"] = task.language
            group.attrs["success"] = success
            group.attrs["planner_model"] = args.model
            group.attrs["privileged_state_used"] = False
        (args.output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        (args.output / "result.json").write_text(json.dumps({
            "suite": args.suite, "task": args.task, "state": args.state, "instruction": task.language,
            "success": success, "steps": len(actions), "planner_model": args.model,
            "inputs": ["language", "RGB-D", "camera calibration", "proprioception"],
            "forbidden_inputs_used": [], "qwen_used": False,
            "collector_parameters": {"grasp_height_above_table": args.grasp_height_above_table, "place_height_above_table": args.place_height_above_table},
            "artifacts": ["plan.json", "trace.json", "trajectory.hdf5"],
        }, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
