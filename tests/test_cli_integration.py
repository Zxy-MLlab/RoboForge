import json
import os
from pathlib import Path
import subprocess
import sys


def test_cli_runs_real_function_calling_controller_loop(tmp_path):
    plugin = tmp_path / "plugin.py"
    plugin.write_text('''
import json
class Adapter:
    instruction = "set marker"
    def __init__(self, task=None, root=None): self.value = 0
    def dispatch(self, method, arguments):
        if method == "act": self.value = arguments["action"]["value"]; return {"ok": True}
        if method == "verify": return {"verified": self.value == 1}
        if method == "observe": return {"value": self.value}
        if method == "record": return {"recorded": True}
        if method == "use": return {"result": {}}
        raise ValueError(method)
    def project_rpc_output(self, method, arguments, result): return dict(result)
    def sensor_report(self, execution): return {"success": self.value == 1}
    def close(self): pass

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

    doctor = subprocess.run([sys.executable, "-m", "embodied_codex", "doctor",
        "--adapter", "plugin:Adapter", "--task", "doctor", "--run-dir", str(tmp_path / "doctor")],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert doctor.returncode == 0, doctor.stdout
    report = json.loads(doctor.stdout)
    assert report["adapter_smoke"] == "available"
    assert report["command_smoke"] == "available"
