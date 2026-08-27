"""Environment-neutral case routing controlled explicitly by the model."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .agent_loop import AgentLoop


class CampaignAdapter:
    """Route one canonical Controller workspace to a model-selected Adapter case.

    The adapter only supplies list/select/run mechanics. It does not order
    cases, focus failures, schedule retries, or decide regression coverage.
    """

    def __init__(self, cases: Sequence[tuple[str, Any]]):
        if not cases:
            raise ValueError("a Campaign requires at least one Adapter case")
        internal_names = [str(name) for name, _adapter in cases]
        if len(set(internal_names)) != len(cases):
            raise ValueError("Campaign case identifiers must be unique")
        self._cases = {f"case-{index:03d}": adapter
                       for index, (_name, adapter) in enumerate(cases, 1)}
        self._evaluator_cases = tuple((str(name), adapter) for name, adapter in cases)
        self.case_ids = tuple(self._cases)
        self.active_case = self.case_ids[0]
        first = self._cases[self.active_case]
        self.sdk_index = getattr(first, "sdk_index", None) or getattr(first, "sdk_contract", None)
        self._instruction = str(getattr(first, "instruction", ""))

    @property
    def active(self):
        return self._cases[self.active_case]

    @property
    def instruction(self):
        return self._instruction

    @property
    def artifact_dir(self):
        return getattr(self.active, "artifact_dir", None)

    @property
    def artifact_roots(self):
        return [getattr(adapter, "artifact_dir") for adapter in self._cases.values()
                if getattr(adapter, "artifact_dir", None)]

    def case_adapters(self):
        """Expose case membership to external evaluation only."""
        return self._evaluator_cases

    def select(self, case_id: str) -> None:
        case_id = str(case_id)
        if case_id not in self._cases:
            raise KeyError(case_id)
        self.active_case = case_id

    def initial_observation(self):
        return self.active.initial_observation()

    def dispatch(self, method, arguments):
        return self.active.dispatch(method, arguments)

    def project_rpc_output(self, method, arguments, result):
        return self.active.project_rpc_output(method, arguments, result)

    def canonical_embodied_state(self):
        provider = getattr(self.active, "canonical_embodied_state", None)
        if not callable(provider):
            raise RuntimeError("active Adapter must provide canonical_embodied_state")
        value = provider()
        if not isinstance(value, Mapping):
            raise RuntimeError("Adapter canonical_embodied_state must return a mapping")
        return value

    def canonical_observation(self, observation):
        provider = getattr(self.active, "canonical_observation", None)
        if not callable(provider):
            raise RuntimeError("active Adapter must provide canonical_observation")
        value = provider(observation)
        if not isinstance(value, Mapping):
            raise RuntimeError("Adapter canonical_observation must return a mapping")
        return value

    def project_public_entities(self, tool_id, result):
        provider = getattr(self.active, "project_public_entities", None)
        return provider(tool_id, result) if callable(provider) else []

    def sensor_report(self, execution):
        return self.active.sensor_report(execution)

    def agent_evidence(self, execution, sensor_report):
        provider = getattr(self.active, "agent_evidence", None)
        return provider(execution, sensor_report) if callable(provider) else {}

    def verification_receipt(self, execution):
        return self.active.verification_receipt(execution)

    def execution_identity(self):
        return self.active.execution_identity()

    def resume_protocol(self):
        return self.active.resume_protocol()

    def validate_execution_receipt(self, receipt):
        return self.active.validate_execution_receipt(receipt)

    def validate_historical_receipt(self, identity, receipt):
        for adapter in self._cases.values():
            if adapter.execution_identity() == identity:
                return adapter.validate_execution_receipt(receipt)
        return False

    def reset_case(self):
        reset = getattr(self.active, "reset_case", None)
        if not callable(reset):
            reset = getattr(self.active, "restart_episode", None)
        if not callable(reset):
            raise RuntimeError("active Adapter does not support reset_case")
        return reset()

    def register_capability(self, tool_id, function, contract):
        # Binding is a deployment mechanism, not a test-order strategy. A Tool
        # selected by the model is made available to every selectable case.
        for adapter in self._cases.values():
            adapter.register_capability(tool_id, function, contract)

    def native_capability_manifest(self):
        manifests = []
        for adapter in self._cases.values():
            provider = getattr(adapter, "native_capability_manifest", None)
            manifests.append(dict(provider() or {}) if callable(provider) else {})
        if any(value != manifests[0] for value in manifests[1:]):
            raise RuntimeError("Campaign cases expose different native capability contracts")
        return manifests[0]

    def resolve_controller_artifact(self, handle: str):
        resolver = getattr(self.active, "resolve_controller_artifact", None)
        if not callable(resolver):
            raise RuntimeError("active Adapter does not expose an artifact resolver")
        return resolver(handle)

    def close(self):
        errors = []
        for adapter in self._cases.values():
            try:
                adapter.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"{len(errors)} Campaign Adapter case(s) failed to close") from errors[0]


class CampaignRunner(AgentLoop):
    """Thin compatibility entry point for an explicitly selected case set."""

    adapter: CampaignAdapter

    def __init__(self, **kwargs):
        if not isinstance(kwargs.get("adapter"), CampaignAdapter):
            raise TypeError("CampaignRunner requires CampaignAdapter")
        super().__init__(**kwargs)

    def run(self, task: str | None = None):
        result = super().run(task)
        return {**result,
                "available_cases": list(self.adapter.case_ids),
                "selected_case": self.adapter.active_case}


__all__ = ["CampaignAdapter", "CampaignRunner"]
