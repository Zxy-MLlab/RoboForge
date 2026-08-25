from .policy import BenchmarkPolicy


class ProvenancePolicy(BenchmarkPolicy):
    def before_run(self, loop):
        loop.event_store.commit("evaluation_policy", {"policy": self.name, "phase": "before_run"})

    def after_run(self, loop, result):
        missing = []
        for tool_id in getattr(loop.capability_manager, "_bound", {}):
            manifest = loop.capability_manager.tool_library.inspect(tool_id)["manifest"]
            if manifest.get("source_sha256") is None: missing.append(tool_id)
        if missing: raise RuntimeError(f"provenance policy rejected un-hashed Tools: {missing}")
        loop.event_store.commit("evaluation_policy", {"policy": self.name, "phase": "after_run", "passed": True})
