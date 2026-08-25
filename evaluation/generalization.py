from .policy import BenchmarkPolicy


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
