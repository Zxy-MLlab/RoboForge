"""Run bounded Embodied Codex breadth-smokes across independent LIBERO tasks.

This launcher deliberately keeps one persistent workspace per task while sharing
one immutable Capability Library across the matrix.  A task-level physical
failure is data, not a reason to abort the remaining Harness checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Sequence

from embodied_codex.conformance import audit_run


def build_task_command(args: argparse.Namespace, task: int, run_dir: Path,
                       capability_root: Path) -> list[str]:
    command=[sys.executable,"-m","embodied_codex.examples.run_libero",
             "--run-dir",str(run_dir),"--suite",args.suite,
             "--task",str(task),"--max-iterations",str(args.max_iterations_per_task),
             "--device",args.device,"--model",args.model,
             "--reasoning-effort",args.reasoning_effort,
             "--base-url",args.base_url,"--config",args.config,
             "--capability-library",str(capability_root),"--states"]
    command.extend(str(state) for state in args.states)
    if args.retry_locked_validation:command.append("--retry-locked-validation")
    return command


def _state_summary(run_dir: Path) -> dict:
    path=run_dir/"state.json"
    if not path.is_file():return {"status":"not_started","iterations":0}
    try:state=json.loads(path.read_text())
    except Exception as exc:
        return {"status":"invalid_state","iterations":0,
                "error":f"{type(exc).__name__}: {exc}"}
    iterations=state.get("iterations") or []
    return {"status":state.get("status"),"iterations":len(iterations),
            "robot_episodes":sum(row.get("evidence") is not None for row in iterations),
            "sensor_task_success":state.get("status")=="sensor_success",
            "frozen_skill":state.get("skill")}


def run_matrix(args: argparse.Namespace,
               runner: Callable[...,subprocess.CompletedProcess]=subprocess.run) -> dict:
    root=Path(args.output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    capability_root=(Path(args.capability_library).resolve()
                     if args.capability_library else root/"shared_capabilities")
    capability_root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for task in args.tasks:
        run_dir=root/f"{args.suite}_task_{task:02d}"
        run_dir.mkdir(parents=True,exist_ok=True)
        command=build_task_command(args,task,run_dir,capability_root)
        log_path=run_dir/"matrix_process.log"
        started=time.time()
        print(f"[matrix] task={task} start run={run_dir}",flush=True)
        with log_path.open("a") as log:
            log.write(f"\n=== matrix launch {started:.6f} ===\n")
            log.write(json.dumps(command)+"\n");log.flush()
            try:
                process=runner(command,stdout=log,stderr=subprocess.STDOUT,
                               text=True,check=False)
                returncode=int(process.returncode);launch_error=None
            except Exception as exc:
                returncode=1;launch_error=f"{type(exc).__name__}: {exc}"
                log.write(launch_error+"\n")
        # Audit even a crashed task: missing gates are useful Harness evidence.
        audit=audit_run(run_dir)
        row={"task_index":task,"run_dir":str(run_dir),"command":command,
             "returncode":returncode,"launch_error":launch_error,
             "duration_seconds":time.time()-started,"process_log":str(log_path),
             "task_result":_state_summary(run_dir),"harness_audit":audit}
        rows.append(row)
        summary={"protocol":"embodied-codex-libero-conformance-matrix-v1",
                 "suite":args.suite,"tasks":list(args.tasks),"states":list(args.states),
                 "max_iterations_per_task":args.max_iterations_per_task,
                 "shared_capability_root":str(capability_root),"runs":rows,
                 "metrics":{
                   "tasks_attempted":len(rows),
                   "sensor_task_successes":sum(r["task_result"].get("sensor_task_success",False)
                                               for r in rows),
                   "harness_conformant_runs":sum(r["harness_audit"]["conformant"] for r in rows),
                   "process_failures":sum(r["returncode"] not in (0,2) for r in rows)}}
        (root/"matrix_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(f"[matrix] task={task} rc={returncode} "
              f"sensor_success={row['task_result'].get('sensor_task_success',False)} "
              f"conformant={audit['conformant']}",flush=True)
    return summary


def parse_args(argv: Sequence[str]|None=None) -> argparse.Namespace:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--suite",default="libero_spatial")
    parser.add_argument("--tasks",type=int,nargs="+",required=True)
    parser.add_argument("--states",type=int,nargs="+",default=[0])
    parser.add_argument("--max-iterations-per-task",type=int,default=2)
    parser.add_argument("--capability-library")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--model",default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort",default="high")
    parser.add_argument("--base-url",default="https://api.apexin.ai/v1")
    parser.add_argument("--config",default="config/standalone_libero")
    parser.add_argument("--retry-locked-validation",action="store_true")
    args=parser.parse_args(argv)
    if len(set(args.tasks))!=len(args.tasks):parser.error("tasks must be unique")
    if len(set(args.states))!=len(args.states):parser.error("states must be unique")
    if args.max_iterations_per_task<1:parser.error("iteration budget must be positive")
    return args


def main(argv: Sequence[str]|None=None) -> int:
    summary=run_matrix(parse_args(argv))
    print(json.dumps(summary["metrics"],indent=2),flush=True)
    # Only Harness / process failures affect this smoke launcher's status.
    # Robot inability is an experimental result and remains visible separately.
    return 0 if summary["metrics"]["process_failures"]==0 else 1


if __name__=="__main__":raise SystemExit(main())
