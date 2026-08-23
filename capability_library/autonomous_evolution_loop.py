"""Persistent, evaluator-blind controller evolution for embodied environments.

The loop owns iteration state, while an environment-specific authoring callback
creates and executes one immutable controller candidate per round.  Only
sensor evidence is retained in the loop state; evaluator-only keys are removed
recursively before persistence.  The callback can be backed by Thea/GPT or a
different agent implementation, so the orchestration is environment agnostic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


_HIDDEN_KEYS = {
    "success", "reward", "done", "terminated", "truncated",
    "check_success", "evaluator_result", "evaluator_success",
}


def sensor_only(value: Any) -> Any:
    """Remove evaluator-labelled fields before an agent state is persisted."""
    if isinstance(value, dict):
        return {
            str(key): sensor_only(item)
            for key, item in value.items()
            if str(key).casefold() not in _HIDDEN_KEYS
        }
    if isinstance(value, list):
        return [sensor_only(item) for item in value]
    return value


@dataclass(frozen=True)
class EvolutionConfig:
    task: str
    max_rounds: int = 8
    stop_on_sensor_success: bool = True
    acquisition_after_same_failure: int = 2
    max_authoring_attempts_per_round: int = 3
    force_acquisition_next_round: bool = False

    def __post_init__(self) -> None:
        if not str(self.task).strip():
            raise ValueError("task must not be empty")
        if not 1 <= int(self.max_rounds) <= 100:
            raise ValueError("max_rounds must be within [1, 100]")
        if not 1 <= int(self.acquisition_after_same_failure) <= 10:
            raise ValueError("acquisition_after_same_failure must be within [1, 10]")
        if not 1 <= int(self.max_authoring_attempts_per_round) <= 10:
            raise ValueError("max_authoring_attempts_per_round must be within [1, 10]")


class AutonomousEvolutionLoop:
    """Run bounded, resumable authoring rounds without evaluator feedback."""

    def __init__(
        self,
        config: EvolutionConfig,
        *,
        state_path: str | Path,
        author_round: Callable[[int, Mapping[str, Any]], Mapping[str, Any]],
        acquire_capabilities: Callable[[int, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        sensor_success: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> None:
        self.config = config
        self.state_path = Path(state_path)
        self.author_round = author_round
        self.acquire_capabilities = acquire_capabilities
        self.sensor_success = sensor_success or self._default_sensor_success

    @staticmethod
    def _default_sensor_success(evidence: Mapping[str, Any]) -> bool:
        return evidence.get("sensor_only_conclusion") == "sensor_verification_passed"

    def _load(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {
                "protocol": "embodied-autonomous-evolution-v1",
                "task": self.config.task,
                "rounds": [],
                "acquisitions": [],
                "authoring_failures": [],
                "status": "not_started",
                "evaluator_visible_to_agent": False,
            }
        state = json.loads(self.state_path.read_text())
        if state.get("task") != self.config.task:
            raise ValueError("existing evolution state belongs to another task")
        return sensor_only(state)

    def _save(self, state: Mapping[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(sensor_only(state), indent=2) + "\n")
        temporary.replace(self.state_path)

    @staticmethod
    def _capability_outcomes(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for event in evidence.get("capability_hook_invocations") or ():
            if not isinstance(event, Mapping):
                continue
            key = (str(event.get("hook") or ""), str(event.get("tool_id") or ""))
            if all(key):
                grouped.setdefault(key, []).append(event)
        stage_signal = {
            "grasp_retry_ranking": bool(evidence.get("attachment_verified")),
            "grasp_execution_profile": bool(evidence.get("attachment_verified")),
            "transport_profile": bool(evidence.get("transport_verified")),
            "support_relation_profile": bool(evidence.get("placement_verified")),
        }
        articulation_verified = any(
            isinstance(item, Mapping)
            and item.get("kind") == "articulation"
            and bool(item.get("verified"))
            for item in evidence.get("verifications") or ()
        )
        outcomes = []
        for (hook, tool_id), events in sorted(grouped.items()):
            applied = sum(bool(item.get("applied")) for item in events)
            generic_stage = str(events[-1].get("stage") or "") if events else ""
            progressed = stage_signal.get(hook, False)
            if hook == "generic_capability":
                progressed = (
                    articulation_verified if generic_stage.startswith("articulation") else
                    bool(evidence.get("attachment_verified"))
                    if generic_stage in {"grasp", "attachment"} else
                    bool(evidence.get("transport_verified"))
                    if generic_stage == "transport" else
                    bool(evidence.get("placement_verified"))
                    if generic_stage in {"placement", "support_relation"} else False
                )
            outcomes.append({
                "hook": hook,
                "tool_id": tool_id,
                **({"stage": generic_stage} if generic_stage else {}),
                "invocations": len(events),
                "applied_invocations": applied,
                "outcome": (
                    "not_applied" if not applied else
                    "stage_progressed" if progressed else
                    "failure_persisted"
                ),
            })
        return outcomes

    @classmethod
    def _failure_history(
        cls, rounds: list[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return compact, evaluator-blind history suitable for an LLM prompt."""
        history = []
        for record in rounds:
            evidence = record.get("sensor_evidence") or {}
            history.append({
                "round": record.get("round"),
                "controller_id": record.get("controller_id"),
                "sensor_only_conclusion": evidence.get("sensor_only_conclusion"),
                "diagnostic_failure_class": cls._effective_failure_class(evidence),
                "phase_diagnostics": evidence.get("phase_diagnostics"),
                "attachment_verified": evidence.get("attachment_verified"),
                "transport_verified": evidence.get("transport_verified"),
                "placement_verified": evidence.get("placement_verified"),
                "correction_status": evidence.get("correction_status"),
                "capability_outcomes": cls._capability_outcomes(evidence),
            })
        return sensor_only(history)

    @staticmethod
    def _acquisition_history(acquisitions: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        history = []
        for record in acquisitions:
            result = record.get("result") or {}
            history.append({
                "before_round": record.get("before_round"),
                "failure_class": record.get("failure_class"),
                "hook_tests": [
                    {
                        "tool_id": item.get("tool_id"),
                        "hook": item.get("hook"),
                        "reason": item.get("reason"),
                    }
                    for item in result.get("hook_tests") or ()
                    if isinstance(item, Mapping)
                ],
                "generic_tests": [
                    {
                        "tool_id": item.get("tool_id"),
                        "reason": item.get("reason"),
                    }
                    for item in result.get("generic_tests") or ()
                    if isinstance(item, Mapping)
                ],
                "implementation_attempted": bool(
                    result.get("implementation_attempted")
                ),
            })
        return sensor_only(history)

    @staticmethod
    def _stage_score(record: Mapping[str, Any]) -> tuple[int, int, int]:
        evidence = record.get("sensor_evidence") or {}
        conclusion = str(evidence.get("sensor_only_conclusion") or "")
        articulation_verified = any(
            isinstance(item, Mapping)
            and item.get("kind") == "articulation"
            and bool(item.get("verified"))
            for item in evidence.get("verifications") or ()
        )
        return (
            5 if conclusion == "sensor_verification_passed" else
            4 if bool(evidence.get("placement_verified")) else
            3 if bool(evidence.get("transport_verified")) else
            2 if bool(evidence.get("attachment_verified")) else
            1 if articulation_verified else 0,
            1 if bool(evidence.get("execution_completed")) else 0,
            int(record.get("round") or 0),
        )

    @classmethod
    def _best_prior(cls, rounds: list[Mapping[str, Any]]) -> dict[str, Any]:
        candidates = [
            item for item in rounds
            if item.get("controller_id") and item.get("sensor_evidence")
        ]
        return dict(max(candidates, key=cls._stage_score)) if candidates else {}

    @staticmethod
    def _effective_failure_class(evidence: Mapping[str, Any]) -> str | None:
        """Upgrade older coarse evidence using persisted legal phase summaries."""
        diagnosed = str(evidence.get("diagnostic_failure_class") or "")
        if diagnosed and diagnosed != "development_run_completed_without_verification":
            return diagnosed
        phases = ((evidence.get("phase_diagnostics") or {}).get("phases") or {})
        candidates = []
        for phase, summary in phases.items():
            if not isinstance(summary, Mapping):
                continue
            commands = int(summary.get("commands") or 0)
            reached = int(summary.get("reached") or 0)
            if commands >= 5 and reached / commands < 0.20:
                candidates.append((int(summary.get("last_command_index") or 0), str(phase)))
        if candidates:
            phase = max(candidates)[1].casefold()
            normalized = (
                "contact" if "contact" in phase else
                "approach" if "approach" in phase else
                "transport" if "transport" in phase else "motion"
            )
            return f"{normalized}_convergence_failed"
        return diagnosed or str(evidence.get("sensor_only_conclusion") or "") or None

    @staticmethod
    def _same_failure_streak(rounds: list[Mapping[str, Any]]) -> tuple[str | None, int]:
        if not rounds:
            return None, 0
        latest_evidence = rounds[-1].get("sensor_evidence") or {}
        conclusion = AutonomousEvolutionLoop._effective_failure_class(latest_evidence)
        if not conclusion or conclusion == "sensor_verification_passed":
            return None, 0
        count = 0
        for record in reversed(rounds):
            evidence = record.get("sensor_evidence") or {}
            current = AutonomousEvolutionLoop._effective_failure_class(evidence)
            if current != conclusion:
                break
            count += 1
        return str(conclusion), count

    def _round_context(
        self,
        rounds: list[Mapping[str, Any]],
        acquisition: Mapping[str, Any] | None,
        acquisitions: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        prior = dict(rounds[-1]) if rounds else {}
        context = {
            # Keep the legacy top-level fields for environment adapters while
            # also exposing the full compact history to the reasoning model.
            **prior,
            "prior_round": prior,
            "best_prior": self._best_prior(rounds),
            "failure_history": self._failure_history(rounds),
            # Authoring may be retried after a process restart. Preserve the
            # most recent completed acquisition instead of making the newly
            # tested Tool disappear merely because this invocation did not
            # trigger another search session.
            "latest_acquisition": dict(
                acquisition or (acquisitions[-1] if acquisitions else {})
            ),
            "acquisition_history": self._acquisition_history(acquisitions),
        }
        return sensor_only(context)

    def run(self) -> dict[str, Any]:
        state = self._load()
        rounds = list(state.get("rounds") or [])
        acquisitions = list(state.get("acquisitions") or [])
        authoring_failures = list(state.get("authoring_failures") or [])
        if state.get("status") == "sensor_success":
            return state
        # A persisted run may be resumed with a larger explicit round budget.
        # Existing immutable rounds remain untouched.
        if state.get("status") == "exhausted" and len(rounds) >= self.config.max_rounds:
            return state
        state["status"] = "running"
        self._save(state)
        first_pending_round = len(rounds) + 1
        for round_id in range(len(rounds) + 1, self.config.max_rounds + 1):
            acquisition = None
            failure_class, streak = self._same_failure_streak(rounds)
            trigger_after_round = rounds[-1].get("round") if rounds else None
            already_acquired = any(
                item.get("trigger_after_round") == trigger_after_round
                for item in acquisitions
            )
            forced_acquisition = bool(
                self.config.force_acquisition_next_round
                and round_id == first_pending_round
            )
            if (
                self.acquire_capabilities is not None
                and failure_class is not None
                and (
                    forced_acquisition
                    or (
                        streak >= self.config.acquisition_after_same_failure
                        and not already_acquired
                    )
                )
            ):
                request = sensor_only({
                    "failure_class": failure_class,
                    "consecutive_failures": streak,
                    "trigger_after_round": trigger_after_round,
                    "forced_after_harness_upgrade": forced_acquisition,
                    "failure_history": self._failure_history(rounds),
                    "latest_sensor_evidence": (rounds[-1].get("sensor_evidence") or {}),
                    "capability_outcomes": self._capability_outcomes(
                        rounds[-1].get("sensor_evidence") or {}
                    ),
                    "acquisition_history": self._acquisition_history(acquisitions),
                })
                try:
                    acquired = dict(self.acquire_capabilities(round_id, request))
                    acquisition = {
                        "before_round": round_id,
                        "trigger_after_round": trigger_after_round,
                        "failure_class": failure_class,
                        "consecutive_failures": streak,
                        "forced_after_harness_upgrade": forced_acquisition,
                        "result": sensor_only(acquired),
                    }
                except Exception as exc:
                    acquisition = {
                        "before_round": round_id,
                        "trigger_after_round": trigger_after_round,
                        "failure_class": failure_class,
                        "consecutive_failures": streak,
                        "forced_after_harness_upgrade": forced_acquisition,
                        "result": {
                            "acquisition_completed": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    }
                acquisitions.append(acquisition)
                state["acquisitions"] = acquisitions
                self._save(state)
            context = self._round_context(rounds, acquisition, acquisitions)
            result = None
            for authoring_attempt in range(
                1, self.config.max_authoring_attempts_per_round + 1
            ):
                try:
                    result = dict(self.author_round(round_id, context))
                    break
                except Exception as exc:
                    authoring_failures.append(sensor_only({
                        "round": round_id,
                        "attempt": authoring_attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "consumed_as_experiment_round": False,
                    }))
                    state["authoring_failures"] = authoring_failures
                    state["status"] = "authoring_retry"
                    self._save(state)
                    if authoring_attempt >= self.config.max_authoring_attempts_per_round:
                        state["status"] = "authoring_error"
                        self._save(state)
                        raise
            assert result is not None
            evidence = sensor_only(result.get("sensor_evidence") or {})
            record = {
                "round": round_id,
                "started_unix": time.time(),
                "controller_id": result.get("controller_id"),
                "sensor_evidence": evidence,
                "tool_events": sensor_only(result.get("tool_events") or []),
                "asset_events": sensor_only(result.get("asset_events") or []),
                "trace_path": result.get("trace_path"),
                "sensor_success": bool(self.sensor_success(evidence)),
            }
            rounds.append(record)
            state["rounds"] = rounds
            state["last_round"] = round_id
            state["status"] = "sensor_success" if record["sensor_success"] and self.config.stop_on_sensor_success else "running"
            self._save(state)
            if record["sensor_success"] and self.config.stop_on_sensor_success:
                return state
        state["status"] = "exhausted"
        self._save(state)
        return state


__all__ = ["AutonomousEvolutionLoop", "EvolutionConfig", "sensor_only"]
