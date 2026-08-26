from .policy import BenchmarkPolicy


class SealedEvaluationPolicy(BenchmarkPolicy):
    def evaluate_frozen(self, *, adapter, runtime, controller, hidden_evaluator=None,
                        dependency_resolver=None):
        """Evaluate frozen I/O with an independent hidden evaluator.

        The coding model is intentionally absent from this path.  Adapter
        sensor receipts are diagnostics only; benchmark truth must come from
        the evaluator barrier (for LIBERO, ``env.check_success``).
        """
        if callable(dependency_resolver):
            dependency_resolver(adapter)
        targets = (adapter.case_adapters() if callable(getattr(adapter, "case_adapters", None))
                   else (("default", adapter),))
        reports = []
        for case_id, case in targets:
            reset = getattr(case, "reset_case", None)
            if callable(reset):
                reset()
            begin = getattr(case, "begin_controller_execution", None)
            if callable(begin):
                begin()
            execution = runtime.execute(controller, case)
            seal = getattr(case, "seal_controller_execution", None)
            if callable(seal):
                seal()
            receipt_provider = getattr(case, "verification_receipt", None)
            receipt = dict(receipt_provider(execution)) if callable(receipt_provider) else {}
            case_evaluator = hidden_evaluator
            if case_evaluator is None:
                case_evaluator = (getattr(case, "hidden_evaluator", None)
                                  or getattr(case, "sealed_evaluator", None)
                                  or getattr(case, "_sealed_check_once", None))
            if not callable(case_evaluator):
                raise RuntimeError("sealed evaluation requires an independent hidden evaluator")
            evaluator_result = (case_evaluator(execution, case)
                                if _accepts_two(case_evaluator) else case_evaluator(execution))
            if isinstance(evaluator_result, dict):
                evaluator_success = evaluator_result.get("success") is True
            else:
                evaluator_success = evaluator_result is True
            reports.append({"case": str(case_id), "execution_status": "completed" if execution.get("completed") is True else "failed",
                            "passed": evaluator_success, "evaluator_success": evaluator_success,
                            "controller_sha256": receipt.get("controller_sha256") or execution.get("program_sha256")})
        successes = sum(1 for row in reports if row["evaluator_success"])
        return {"sealed_evaluation": bool(reports) and successes == len(reports),
                "sealed_evaluation_cases": reports,
                "episodes": len(reports), "evaluator_successes": successes,
                "success_rate": successes / len(reports) if reports else 0.0,
                "controller_sha256": reports[0].get("controller_sha256") if reports else None,
                "evaluation_passed": bool(reports) and successes == len(reports)}

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
            passed = report is True
            reports.append({"case": str(case_id), "passed": passed})
        passed = bool(reports) and all(row["passed"] for row in reports)
        result["sealed_evaluation"] = passed
        result["sealed_evaluation_cases"] = reports
        result["episodes"] = len(reports)
        result["evaluator_successes"] = sum(1 for row in reports if row["passed"])
        result["success_rate"] = result["evaluator_successes"] / len(reports) if reports else 0.0
        result["evaluation_passed"] = passed
        loop.event_store.commit("evaluation_policy", {"policy": self.name,
            "phase": "after_run", "result": passed, "cases": reports})


def _accepts_two(callback):
    try:
        import inspect
        return len(inspect.signature(callback).parameters) >= 2
    except (TypeError, ValueError):
        return False
