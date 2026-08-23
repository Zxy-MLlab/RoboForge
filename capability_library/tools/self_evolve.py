"""Self-evolution controller for an internet-enabled embodied Harness.

The controller owns the improvement loop, while an LLM supplies failure
hypotheses and chooses which public leads to inspect. Sealed benchmark results
are intentionally not accepted as input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from asset_provenance_gate import evaluate_asset_manifest
from public_resource_market import record_acquisition_event, search_public_resources


@dataclass(frozen=True)
class EvolutionRound:
    round_id: int
    failure_summary: str
    search_queries: tuple[str, ...]
    discovered_resources: tuple[dict[str, Any], ...]
    accepted_assets: tuple[str, ...]
    rejected_assets: tuple[dict[str, Any], ...]
    integration_results: tuple[dict[str, Any], ...] = ()
    retry_result: dict[str, Any] | None = None
    development_score_before: float | None = None
    development_score_after: float | None = None


@dataclass
class SelfEvolutionController:
    """Run bounded autonomous capability-acquisition rounds on development data."""

    current_tasks: tuple[str, ...]
    ledger_path: str = "artifacts/capability_acquisition.jsonl"
    rounds: list[EvolutionRound] = field(default_factory=list)

    def evolve_round(
        self,
        failure_summary: str,
        search_queries: Sequence[str],
        *,
        asset_manifests: Sequence[Mapping[str, Any]] = (),
        development_score_before: float | None = None,
        development_score_after: float | None = None,
        search_fn: Callable[..., dict[str, Any]] = search_public_resources,
        integrate_fn: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
        retry_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> EvolutionRound:
        """Search, gate, integrate, and retry one development-only round."""
        if not str(failure_summary).strip():
            raise ValueError("failure_summary must not be empty")
        if development_score_before is not None and not 0 <= development_score_before <= 1:
            raise ValueError("development_score_before must be within [0, 1]")
        if development_score_after is not None and not 0 <= development_score_after <= 1:
            raise ValueError("development_score_after must be within [0, 1]")

        resources: list[dict[str, Any]] = []
        normalized_queries = tuple(str(query).strip() for query in search_queries if str(query).strip())
        for query in normalized_queries:
            result = search_fn(query, ledger_path=self.ledger_path)
            if result.get("success"):
                resources.extend(result.get("results", []))

        accepted: list[str] = []
        rejected: list[dict[str, Any]] = []
        integration_results: list[dict[str, Any]] = []
        for manifest in asset_manifests:
            decision = evaluate_asset_manifest(
                manifest,
                current_tasks=self.current_tasks,
            )
            asset_id = str(manifest.get("id") or "")
            if decision["eligible"]:
                accepted.append(asset_id)
                record_acquisition_event(
                    {
                        "event": "asset_accepted_for_development",
                        "asset_id": asset_id,
                        "reporting_stratum": decision["reporting_stratum"],
                        "failure_summary": failure_summary,
                    },
                    ledger_path=self.ledger_path,
                )
                if integrate_fn is not None:
                    try:
                        integration = dict(integrate_fn(manifest))
                    except Exception as exc:
                        integration = {
                            "success": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    integration["asset_id"] = asset_id
                    integration_results.append(integration)
                    record_acquisition_event(
                        {
                            "event": "asset_integration_attempt",
                            "asset_id": asset_id,
                            "success": bool(integration.get("success")),
                            "error": integration.get("error"),
                        },
                        ledger_path=self.ledger_path,
                    )
            else:
                rejected.append({"asset_id": asset_id, "reasons": decision["reasons"]})
                record_acquisition_event(
                    {
                        "event": "asset_rejected_by_provenance_gate",
                        "asset_id": asset_id,
                        "reasons": decision["reasons"],
                        "failure_summary": failure_summary,
                    },
                    ledger_path=self.ledger_path,
                )

        retry_result: dict[str, Any] | None = None
        if retry_fn is not None and any(item.get("success") for item in integration_results):
            try:
                retry_result = dict(retry_fn())
            except Exception as exc:
                retry_result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
            record_acquisition_event(
                {
                    "event": "development_retry",
                    "success": bool(retry_result.get("success")),
                    "result": retry_result,
                },
                ledger_path=self.ledger_path,
            )

        round_record = EvolutionRound(
            round_id=len(self.rounds) + 1,
            failure_summary=str(failure_summary),
            search_queries=normalized_queries,
            discovered_resources=tuple(resources),
            accepted_assets=tuple(accepted),
            rejected_assets=tuple(rejected),
            integration_results=tuple(integration_results),
            retry_result=retry_result,
            development_score_before=development_score_before,
            development_score_after=development_score_after,
        )
        self.rounds.append(round_record)
        return round_record

    def export_state(self, path: str | Path) -> None:
        """Persist the self-evolution trace without evaluator-only outcomes."""
        payload = {
            "current_tasks": list(self.current_tasks),
            "rounds": [
                {
                    "round_id": item.round_id,
                    "failure_summary": item.failure_summary,
                    "search_queries": list(item.search_queries),
                    "discovered_resources": list(item.discovered_resources),
                    "accepted_assets": list(item.accepted_assets),
                    "rejected_assets": list(item.rejected_assets),
                    "integration_results": list(item.integration_results),
                    "retry_result": item.retry_result,
                    "development_score_before": item.development_score_before,
                    "development_score_after": item.development_score_after,
                }
                for item in self.rounds
            ],
            "sealed_results_consumed": False,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n")


def register_self_evolution_tool(
    registry: Any,
    controller: SelfEvolutionController,
    *,
    state_path: str | Path | None = None,
    integrate_fn: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    retry_fn: Callable[[], dict[str, Any]] | None = None,
) -> None:
    """Expose one bounded development-only self-evolution round to Thea."""

    @registry.tool(
        name="self_evolve_from_failure",
        description=(
            "Search public resources, provenance-gate candidate assets, and "
            "record one self-evolution round using development failures only."
        ),
    )
    def self_evolve_from_failure(
        failure_summary: str,
        search_queries: list[str],
        asset_manifests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        round_record = controller.evolve_round(
            failure_summary,
            search_queries,
            asset_manifests=asset_manifests,
            integrate_fn=integrate_fn,
            retry_fn=retry_fn,
        )
        if state_path is not None:
            controller.export_state(state_path)
        return {
            "success": True,
            "round_id": round_record.round_id,
            "discovered_resources": len(round_record.discovered_resources),
            "accepted_assets": list(round_record.accepted_assets),
            "rejected_assets": list(round_record.rejected_assets),
            "integration_results": list(round_record.integration_results),
            "retry_result": round_record.retry_result,
            "sealed_results_consumed": False,
        }


__all__ = ["EvolutionRound", "SelfEvolutionController", "register_self_evolution_tool"]
