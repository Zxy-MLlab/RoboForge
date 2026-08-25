from .policy import BenchmarkPolicy


class AntiCheatingPolicy(BenchmarkPolicy):
    def before_run(self, loop):
        loop.event_store.commit("evaluation_policy", {"policy": self.name, "phase": "before_run"})

    def after_run(self, loop, result):
        for event in loop.event_store.events():
            if event.get("kind") == "execution":
                payload = event.get("payload", {})
                if payload.get("sensor_report", {}).get("benchmark_signal_exposed"):
                    raise RuntimeError("anti-cheating policy detected benchmark signal")
        loop.event_store.commit("evaluation_policy", {"policy": self.name, "phase": "after_run", "passed": True})
