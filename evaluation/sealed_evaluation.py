from .policy import BenchmarkPolicy


class SealedEvaluationPolicy(BenchmarkPolicy):
    def before_run(self, loop):
        loop.event_store.commit("evaluation_policy", {"policy": self.name, "phase": "before_run"})

    def after_run(self, loop, result):
        seal = getattr(loop.adapter, "seal_controller_execution", None)
        evaluate = getattr(loop.adapter, "_sealed_check_once", None)
        if callable(seal): seal()
        report = evaluate() if callable(evaluate) else None
        result["sealed_evaluation"] = report
        loop.event_store.commit("evaluation_policy", {"policy": self.name, "phase": "after_run", "result": report})
