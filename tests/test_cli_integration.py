import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys


def test_cli_runs_real_function_calling_controller_loop(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text('''
import json
from pathlib import Path
class Adapter:
    instruction = "set marker"
    def __init__(self, task=None, root=None):
        self.value = 0; self.generation = "cli-fake-1"; self.root = Path(root)
    def dispatch(self, method, arguments):
        if method == "act": self.value = arguments["action"]["value"]; return {"ok": True}
        if method == "verify": return {"verified": self.value == 1}
        if method == "observe": return {"value": self.value}
        if method == "record": return {"recorded": True}
        if method == "use": return {"result": {}}
        raise ValueError(method)
    def initial_observation(self): return {"value": self.value}
    def project_rpc_output(self, method, arguments, result): return dict(result)
    def register_capability(self, tool_id, function, contract): pass
    def sensor_report(self, execution): return {"success": self.value == 1}
    def execution_identity(self): return {"episode_id": "cli-fake", "environment_generation": self.generation}
    def resume_protocol(self): return {"supports_resume": True, "resume_token": "cli-resume",
        "environment_generation": self.generation, "actions_idempotent": False, "replay_allowed": True}
    def verification_receipt(self, execution): return {"verified": self.value == 1 and execution.get("completed") is True,
        "controller_sha256": execution.get("program_sha256"), "environment_identity": self.execution_identity(),
        "episode_id": "cli-fake", "environment_generation": self.generation}
    def validate_execution_receipt(self, receipt): return receipt.get("verified") is True and receipt.get("environment_identity") == self.execution_identity()
    def close(self): (self.root / "adapter.closed").write_text("closed")

class Model:
    def __init__(self): self.turn = 0
    def decide(self, *, messages, tools):
        self.turn += 1
        def call(name, args): return {"tool_calls": [{"id": str(self.turn), "name": name,
            "arguments": json.dumps(args)}], "content": ""}
        if self.turn == 1: return call("write_file", {"path": "controller.py",
            "content": "def run(robot):\\n    robot.act({'value': 1})\\n    return robot.verify('goal', {})\\n"})
        if self.turn == 2: return call("run_controller", {})
        return call("finish", {"summary": "verified"})
''')
    run_dir = tmp_path / "run"; assets = tmp_path / "assets"
    env = dict(os.environ, PYTHONPATH=f"{tmp_path}:{Path(__file__).parents[1]}")
    command = [sys.executable, "-m", "embodied_codex", "run", "--adapter", "plugin:Adapter",
               "--task", "set marker", "--model", "plugin:Model", "--run-dir", str(run_dir),
               "--asset-root", str(assets), "--max-steps", "6"]
    completed = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=60)
    assert completed.returncode == 0, completed.stdout
    output = json.loads(completed.stdout)
    assert output["finished"] is True
    assert output["executions"] == 1
    assert (run_dir / "checkpoint/state.json").is_file()
    assert (run_dir / "adapter.closed").is_file()

    failed_dir = tmp_path / "failed-before-loop"
    failed = subprocess.run([sys.executable, "-m", "embodied_codex", "run",
        "--adapter", "plugin:Adapter", "--task", "set marker", "--model", "plugin:MissingModel",
        "--run-dir", str(failed_dir), "--asset-root", str(assets)], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert failed.returncode != 0
    assert (failed_dir / "adapter.closed").is_file()

    doctor = subprocess.run([sys.executable, "-m", "embodied_codex", "doctor",
        "--adapter", "plugin:Adapter", "--model", "plugin:Model", "--task", "doctor",
        "--run-dir", str(tmp_path / "doctor")],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert doctor.returncode == 0, doctor.stdout
    report = json.loads(doctor.stdout)
    assert report["adapter_smoke"] == "available"
    assert report["command_smoke"] == "available"

    checkpoint = tmp_path / "weights.bin"; checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checked = subprocess.run([sys.executable, "-m", "embodied_codex", "doctor",
        "--adapter", "plugin:Adapter", "--model", "plugin:Model",
        "--run-dir", str(tmp_path / "doctor-checksum"), "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", digest], env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60)
    assert checked.returncode == 0, checked.stdout
    assert json.loads(checked.stdout)["checkpoint"]["sha256"] == digest
    rejected = subprocess.run([sys.executable, "-m", "embodied_codex", "doctor",
        "--adapter", "plugin:Adapter", "--model", "plugin:Model",
        "--run-dir", str(tmp_path / "doctor-bad-checksum"), "--checkpoint", str(checkpoint),
        "--checkpoint-sha256", "0" * 64], env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60)
    assert rejected.returncode == 1
    assert json.loads(rejected.stdout)["checkpoint"]["reason"] == "checksum mismatch"


def test_adapter_preflight_runs_and_blocks_doctor_and_formal_run(tmp_path):
    plugin = tmp_path / "preflight_plugin.py"
    plugin.write_text('''
import json
import os
from pathlib import Path

def doctor_checks():
    marker = Path(os.environ["PREFLIGHT_MARKER"])
    marker.write_text(marker.read_text() + "checked\\n" if marker.exists() else "checked\\n")
    return {"ok": os.environ.get("PREFLIGHT_OK") == "1", "dependency": "deliberately unavailable"}

class Adapter:
    def __init__(self, task=None, root=None):
        Path(os.environ["ADAPTER_INIT_MARKER"]).write_text("initialized")
''')
    env = dict(os.environ, PYTHONPATH=f"{tmp_path}:{Path(__file__).parents[1]}",
               PREFLIGHT_MARKER=str(tmp_path / "preflight.called"),
               ADAPTER_INIT_MARKER=str(tmp_path / "adapter.initialized"),
               PREFLIGHT_OK="0")
    doctor = subprocess.run([sys.executable, "-m", "embodied_codex", "doctor",
        "--adapter", "preflight_plugin:Adapter", "--model",
        "embodied_codex.fake_adapter:FakeModel", "--run-dir", str(tmp_path / "doctor")],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert doctor.returncode == 1, doctor.stdout
    report = json.loads(doctor.stdout)
    assert report["adapter_preflight"] == {
        "ok": False, "dependency": "deliberately unavailable"}
    assert report["adapter_smoke"].startswith("unavailable: RuntimeError")
    assert not (tmp_path / "adapter.initialized").exists()

    run = subprocess.run([sys.executable, "-m", "embodied_codex", "run",
        "--adapter", "preflight_plugin:Adapter", "--model",
        "embodied_codex.fake_adapter:FakeModel", "--task", "blocked",
        "--run-dir", str(tmp_path / "run")], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert run.returncode != 0
    assert "Adapter preflight failed" in run.stdout
    assert not (tmp_path / "adapter.initialized").exists()
    assert (tmp_path / "preflight.called").read_text().splitlines() == ["checked", "checked"]
