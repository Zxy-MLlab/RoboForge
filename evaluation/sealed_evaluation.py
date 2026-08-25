from .policy import BenchmarkPolicy


class SealedEvaluationPolicy(BenchmarkPolicy):
    def before_run(self, loop):
        loop.event_store.commit("evaluation_policy", {"policy": self.name, "phase": "before_run"})

    def after_run(self, loop, result):
        enumerate_cases = getattr(loop.adapter, "case_adapters", None)
        targets = (enumerate_cases() if callable(enumerate_cases)
                   else (("default", loop.adapter),))
        reports = []
        for case_id, adapter in targets:
            seal = getattr(adapter, "seal_controller_execution", None)
            evaluate = getattr(adapter, "_sealed_check_once", None)
            if callable(seal):
                seal()
            report = evaluate() if callable(evaluate) else None
            reports.append({"case": str(case_id), "passed": report is True})
        passed = bool(reports) and all(row["passed"] for row in reports)
        result["sealed_evaluation"] = passed
        result["sealed_evaluation_cases"] = reports
        result["evaluation_passed"] = passed
        loop.event_store.commit("evaluation_policy", {"policy": self.name,
            "phase": "after_run", "result": passed, "cases": reports})
