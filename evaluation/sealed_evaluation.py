from .policy import BenchmarkPolicy


class SealedEvaluationPolicy(BenchmarkPolicy):
    def evaluate_frozen(self, *, adapter, runtime, controller):
        """Evaluate a frozen Controller without invoking a coding model."""
        targets = (adapter.case_adapters() if callable(getattr(adapter, "case_adapters", None))
                   else (("default", adapter),))
        reports = []
        for case_id, case in targets:
            reset = getattr(case, "reset_case", None)
            if callable(reset):
                reset()
            execution = runtime.execute(controller, case)
            seal = getattr(case, "seal_controller_execution", None)
            if callable(seal):
                seal()
            verifier = getattr(case, "verification_receipt", None)
            receipt = dict(verifier(execution)) if callable(verifier) else {}
            reports.append({"case": str(case_id), "passed": receipt.get("verified") is True,
                            "controller_sha256": receipt.get("controller_sha256")})
        return {"sealed_evaluation": bool(reports) and all(row["passed"] for row in reports),
                "sealed_evaluation_cases": reports,
                "evaluation_passed": bool(reports) and all(row["passed"] for row in reports)}

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
