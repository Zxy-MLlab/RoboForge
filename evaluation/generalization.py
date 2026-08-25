from .policy import BenchmarkPolicy


class FrozenControllerPolicy(BenchmarkPolicy):
    def __init__(self, expected_sha256: str, name: str = "frozen_controller"):
        super().__init__(name=name); self.expected_sha256 = str(expected_sha256)

    def before_run(self, loop):
        loop.workspace.lock_file("controller.py", self.expected_sha256)
        loop.event_store.commit("evaluation_policy", {"policy": self.name,
            "phase": "before_run", "controller_sha256": self.expected_sha256})

    def after_run(self, loop, result):
        actual = result.get("latest_evidence", {}).get("controller_sha256")
        if actual != self.expected_sha256:
            raise RuntimeError("frozen Controller hash was not executed")
        loop.event_store.commit("evaluation_policy", {"policy": self.name,
            "phase": "after_run", "controller_sha256": actual, "passed": True})


class GeneralizationPolicy(BenchmarkPolicy):
    def before_run(self, loop):
        loop.state["evaluation_controller_hashes"] = []
        loop.event_store.commit("evaluation_policy", {"policy": self.name, "phase": "before_run"})

    def after_run(self, loop, result):
        hashes = [event.get("payload", {}).get("controller_sha256") for event in loop.event_store.events()
                  if event.get("kind") == "execution"]
        loop.state["evaluation_controller_hashes"] = [x for x in hashes if x]
        loop.event_store.commit("evaluation_policy", {"policy": self.name, "phase": "after_run",
                                                        "controller_hashes": loop.state["evaluation_controller_hashes"]})
