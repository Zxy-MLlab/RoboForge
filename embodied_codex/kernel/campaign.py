"""Environment-neutral multi-case convergence on one Controller workspace."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .agent_loop import AgentLoop, ProtocolError


class CampaignAdapter:
    """Route one canonical Controller to a selected Adapter case."""

    def __init__(self, cases: Sequence[tuple[str, Any]]):
        if not cases:
            raise ValueError("a Campaign requires at least one Adapter case")
        self._cases = {str(name): adapter for name, adapter in cases}
        if len(self._cases) != len(cases):
            raise ValueError("Campaign case identifiers must be unique")
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
        """Expose case membership to external orchestration such as evaluation."""
        return tuple(self._cases.items())

    @property
    def episode(self):
        return getattr(self.active, "episode", None)

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

    def sensor_report(self, execution):
        return self.active.sensor_report(execution)

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

    def register_capability(self, tool_id, function, contract):
        for adapter in self._cases.values():
            adapter.register_capability(tool_id, function, contract)

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
    """AgentLoop completion gate requiring one SHA to pass every Adapter case."""

    adapter: CampaignAdapter

    def __init__(self, **kwargs):
        if not isinstance(kwargs.get("adapter"), CampaignAdapter):
            raise TypeError("CampaignRunner requires CampaignAdapter")
        super().__init__(**kwargs)
        campaign = self.state.get("campaign")
        if not isinstance(campaign, Mapping) or campaign.get("cases") != list(self.adapter.case_ids):
            campaign = {"cases": list(self.adapter.case_ids),
                        "active_case": self.adapter.case_ids[0],
                        "controller_sha256": None, "queue": list(self.adapter.case_ids),
                        "validated_cases": {}, "failure_focus": None}
        self.state["campaign"] = dict(campaign)
        active = str(self.state["campaign"].get("active_case") or self.adapter.case_ids[0])
        self.adapter.select(active if active in self.adapter.case_ids else self.adapter.case_ids[0])
        if self.latest_evidence is not None:
            identity = self.latest_evidence.get("environment_identity")
            receipt = self.latest_evidence.get("verification_receipt") or {}
            valid = self.adapter.validate_historical_receipt(identity, receipt)
            self.state["restored_evidence_unverified"] = not valid
            campaign_valid = valid and bool(self.state["campaign"].get("all_cases_verified"))
            if campaign_valid:
                records = self.state["campaign"].get("validated_cases") or {}
                digest = self.state["campaign"].get("controller_sha256")
                for case_id in self.adapter.case_ids:
                    record = records.get(case_id) or {}
                    self.adapter.select(case_id)
                    campaign_valid = bool(campaign_valid
                        and record.get("controller_sha256") == digest
                        and record.get("environment_identity") == self.adapter.execution_identity()
                        and self.adapter.validate_execution_receipt(
                            record.get("verification_receipt") or {}))
                self.adapter.select(active if active in self.adapter.case_ids
                                    else self.adapter.case_ids[0])
            if campaign_valid:
                self.state["finished"] = True
                self.state["completion_valid"] = True
                self.state["successful_cases"] = len(self.adapter.case_ids)
            else:
                self.state["finished"] = False
                self.state["completion_valid"] = False

    @staticmethod
    def _unique(values):
        result = []
        for value in values:
            value = str(value)
            if value not in result:
                result.append(value)
        return result

    def _prepare_controller(self, digest: str) -> dict[str, Any]:
        campaign = dict(self.state["campaign"])
        if campaign.get("controller_sha256") != digest:
            old_validated = list((campaign.get("validated_cases") or {}).keys())
            focus = campaign.get("failure_focus") or campaign.get("active_case")
            campaign.update({"controller_sha256": digest, "validated_cases": {},
                "queue": self._unique([focus, *old_validated, *self.adapter.case_ids]),
                "failure_focus": None})
        queue = [case for case in campaign.get("queue", [])
                 if case in self.adapter.case_ids]
        if not queue:
            queue = [case for case in self.adapter.case_ids
                     if case not in (campaign.get("validated_cases") or {})]
        if queue:
            campaign["active_case"] = queue[0]
            self.adapter.select(queue[0])
        campaign["queue"] = queue
        self.state["campaign"] = campaign
        return campaign

    def _run_controller(self):
        if not self.workspace.controller.is_file():
            raise ProtocolError("controller.py does not exist")
        digest = hashlib.sha256(self.workspace.controller.read_bytes()).hexdigest()
        campaign = self._prepare_controller(digest)
        case_id = str(campaign["active_case"])
        evidence = super()._run_controller()
        evidence["campaign_case"] = case_id
        campaign = dict(self.state["campaign"])
        validated = dict(campaign.get("validated_cases") or {})
        queue = list(campaign.get("queue") or [])
        if self._evidence_success(evidence):
            validated[case_id] = {"controller_sha256": digest,
                "artifact_uri": evidence.get("artifact_uri"),
                "environment_identity": evidence.get("environment_identity"),
                "verification_receipt": evidence.get("verification_receipt"),
                "resume_token": evidence.get("resume_token")}
            queue = [value for value in queue if value != case_id]
            campaign["failure_focus"] = None
        else:
            campaign["failure_focus"] = case_id
            queue = self._unique([case_id, *queue])
        campaign["validated_cases"] = validated
        campaign["queue"] = queue
        if queue:
            campaign["active_case"] = queue[0]
        campaign["all_cases_verified"] = (
            set(validated) == set(self.adapter.case_ids)
            and all(item.get("controller_sha256") == digest for item in validated.values()))
        self.state["campaign"] = campaign
        return {**evidence, "campaign": campaign}

    def _finish(self, summary: str):
        if not self.workspace.controller.is_file():
            raise ProtocolError("completion rejected: controller has not been executed")
        digest = hashlib.sha256(self.workspace.controller.read_bytes()).hexdigest()
        campaign = self._prepare_controller(digest)
        validated = dict(campaign.get("validated_cases") or {})
        errors = []
        missing = [case for case in self.adapter.case_ids if case not in validated]
        if missing:
            errors.append(f"cases have no current verification: {missing}")
        previous = self.adapter.active_case
        try:
            for case_id, record in validated.items():
                if record.get("controller_sha256") != digest:
                    errors.append(f"case {case_id} belongs to an older Controller")
                    continue
                self.adapter.select(case_id)
                if self.adapter.execution_identity() != record.get("environment_identity"):
                    errors.append(f"case {case_id} environment identity changed")
                elif not self.adapter.validate_execution_receipt(
                        record.get("verification_receipt") or {}):
                    errors.append(f"case {case_id} Adapter receipt is no longer valid")
        finally:
            self.adapter.select(previous)
        if errors:
            raise ProtocolError("completion rejected: " + "; ".join(errors))
        self.state.update({"finished": True, "completion_valid": True,
                           "completion_summary": str(summary),
                           "successful_cases": len(self.adapter.case_ids)})
        campaign["all_cases_verified"] = True
        self.state["campaign"] = campaign
        return dict(self.state)

    def run(self, task: str | None = None):
        result = super().run(task)
        return {**result, "campaign": self.state.get("campaign"),
                "cases": list(self.adapter.case_ids)}


__all__ = ["CampaignAdapter", "CampaignRunner"]
