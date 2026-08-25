from .policy import BenchmarkPolicy
import json
from pathlib import Path


class AntiCheatingPolicy(BenchmarkPolicy):
    def before_run(self, loop):
        loop.event_store.commit("evaluation_policy", {"policy": self.name, "phase": "before_run"})

    def after_run(self, loop, result):
        for event in loop.event_store.events():
            if event.get("kind") == "execution":
                payload = event.get("payload", {})
                report = {}
                uri = payload.get("artifact_uri")
                if isinstance(uri, str) and uri.startswith("run://"):
                    path = (Path(loop.root) / uri.removeprefix("run://")).resolve()
                    if Path(loop.root).resolve() not in path.parents:
                        raise RuntimeError("execution evidence URI escapes run root")
                    try:
                        evidence = json.loads(path.read_text())
                    except (OSError, json.JSONDecodeError) as exc:
                        raise RuntimeError("execution evidence cannot be audited") from exc
                    report = evidence.get("sensor_report") or {}
                if report.get("benchmark_signal_exposed") is True:
                    raise RuntimeError("anti-cheating policy detected benchmark signal")
        loop.event_store.commit("evaluation_policy", {"policy": self.name, "phase": "after_run", "passed": True})
