"""Live HTTPS capability-acquisition acceptance through Agent function calls.

Run explicitly because it intentionally depends on public GitHub availability:
    python tests/live_acquisition_acceptance.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.kernel.agent_loop import AgentLoop, LoopBudget
from embodied_codex.kernel.assets import CapabilityGapLibrary, CapabilityLibrary
from embodied_codex.kernel.assets import ExperienceLibrary, SkillLibrary
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.workspace import PersistentWorkspace


def _call(turn, name, arguments):
    return {"content": "", "tool_calls": [{"id": f"live-{turn}", "name": name,
        "arguments": json.dumps(arguments)}]}


class LiveAcquisitionModel:
    """Deterministic tool caller; all network/build/runtime work remains real."""

    def __init__(self):
        self.turn = 0

    def decide(self, *, messages, tools):
        self.turn += 1
        source = "source/RoboForge-main"
        calls = {
            1: ("acquire_capability", {"query": "robot perception python github"}),
            2: ("search_web", {"query": "robot perception python github", "limit": 3}),
            3: ("fetch_web_page", {"url": "https://github.com/Zxy-MLlab/RoboForge",
                                   "max_chars": 2000}),
            4: ("record_decision", {
                "goal": "Acquire and validate a public ranking capability, then use it in a Controller.",
                "evidence_refs": [],
                "hypothesis": "The public source contains a reusable deterministic ranking implementation.",
                "decision": "Download, inspect, package, test, and execute the acquired capability.",
                "expected_effect": "The Controller selects value 1 and receives a successful sensor receipt.",
                "uncertainty": "Network availability and upstream archive contents may change."}),
            5: ("download_public_asset", {
                "url": "https://github.com/Zxy-MLlab/RoboForge/archive/refs/heads/main.zip",
                "filename": "downloads/roboforge.zip"}),
            6: ("unpack_public_asset", {"path": "downloads/roboforge.zip",
                                        "destination": "source"}),
            7: ("read_file", {"path": f"{source}/embodied_codex/retrieval.py",
                              "start_line": 1, "end_line": 120}),
            8: ("write_file", {"path": f"{source}/acquired_tool.py", "content":
                "import os\n"
                "import sys\n"
                "sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'embodied_codex'))\n"
                "from retrieval import rank_records\n"
                "def run(payload):\n"
                "    rows=[{'id':'target','text':'set target','value':1},"
                "{'id':'other','text':'other','value':0}]\n"
                "    best=rank_records(payload['query'],rows,text_fields=('text',),"
                "id_field='id',limit=1)[0]\n"
                "    return {'value':best['value']}\n"}),
            9: ("build_capability", {"directory": source, "argv": [sys.executable,
                "-m", "compileall", "-q", "embodied_codex/retrieval.py", "acquired_tool.py"]}),
            10: ("register_capability_package", {"name": "live_acquired_ranker",
                "bundle_path": source,
                "description": "Publicly downloaded ranking algorithm used as target selector",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"], "additionalProperties": False},
                "output_schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                    "required": ["value"], "additionalProperties": False},
                "package_spec": {"kind": "algorithm", "entrypoint": "acquired_tool.py",
                    "accelerator": "cpu", "runtime_requirements": []},
                "consequence": "READ_ONLY",
                "source_urls": ["https://github.com/Zxy-MLlab/RoboForge"]}),
            11: ("test_tool", {"tool_id": "live_acquired_ranker:v001",
                "cases": [{"input": {"query": "set target"}, "expected": {"value": 1}}]}),
            12: ("write_file", {"path": "controller.py", "content":
                "def run(robot):\n"
                "    receipt=robot.use('live_acquired_ranker:v001',{'query':'set target'})\n"
                "    result=receipt['result']\n"
                "    robot.act({'type':'set_value','value':result['value']})\n"
                "    return robot.verify('target',{})\n"}),
            13: ("run_controller", {}),
            14: ("finish", {"summary": "live acquired capability verified"}),
        }
        name, arguments = calls[self.turn]
        return _call(self.turn, name, arguments)


def main():
    root = Path(tempfile.mkdtemp(prefix="roboforge-live-agent-"))
    workspace = PersistentWorkspace(root / "run/workspace")
    adapter = FakeAdapter("acquire target", root / "run")
    tools = CapabilityLibrary(root / "assets/tools", workspace.root,
        python=sys.executable)
    manager = CapabilityManager(asset_root=root / "assets", workspace=workspace, adapter=adapter,
        tool_library=tools, skill_library=SkillLibrary(root / "assets/skills"),
        experience_library=ExperienceLibrary(root / "assets/experiences"),
        gap_library=CapabilityGapLibrary(root / "assets/gaps"))
    loop = AgentLoop(model=LiveAcquisitionModel(), workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
            asset_registry=manager, workspace=workspace), capability_manager=manager,
        runtime=ControllerRuntime(timeout_seconds=30), event_store=EventStore(root / "run"),
        root=root / "run", budget=LoopBudget(max_steps=18, max_executions=3), resume=False)
    try:
        result = loop.run(adapter.instruction)
    finally:
        adapter.close()
    events = EventStore(root / "run").events()
    invoked = [row["payload"]["name"] for row in events if row["kind"] == "tool_result"]
    expected = ["acquire_capability", "search_web", "fetch_web_page", "record_decision", "download_public_asset",
        "unpack_public_asset", "read_file", "write_file", "build_capability",
        "register_capability_package", "test_tool", "write_file", "run_controller", "finish"]
    if invoked != expected or result.get("finished") is not True:
        raise SystemExit(json.dumps({"result": result, "invoked": invoked}, indent=2, default=str))
    execution_event = next(row for row in reversed(events)
                           if row["kind"] == "execution")
    evidence_uri = execution_event["payload"]["artifact_uri"]
    evidence_path = root / "run" / evidence_uri.removeprefix("run://")
    evidence = json.loads(evidence_path.read_text())
    print(json.dumps({"root": str(root), "finished": result["finished"],
        "executions": result["executions"], "tool_calls": invoked,
        "sensor_success": evidence["sensor_report"]["sensor_success"],
        "verification_receipt": evidence["verification_receipt"]},
        indent=2))


if __name__ == "__main__":
    main()
