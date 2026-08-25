from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator, ValidationError


@dataclass
class Function:
    name: str; description: str; parameters: Mapping[str, Any]; call: Callable[..., Any]
    available: Callable[[],bool]|None = None
    evidence_policy: str | Callable[[Mapping[str,Any]],str] = "repeatable"
    evidence_group: str = "default"
    invalidates_evidence_groups: tuple[str,...] = ()
    evidence_progress: bool | Callable[[Mapping[str,Any]],bool] = False
    execution_progress: bool | Callable[[Mapping[str,Any]],bool] = False
    post_mutation_read_allowed: bool | Callable[[Mapping[str,Any]],bool] = False
    @property
    def schema(self):
        return {"type": "function", "function": {"name": self.name,
                "description": self.description, "parameters": dict(self.parameters)}}


class FunctionRegistry:
    def __init__(self): self.items: dict[str, Function] = {}
    def add(self, name, description, parameters, call, available=None, *,
            evidence_policy="repeatable", evidence_group="default",
            invalidates_evidence_groups=(), evidence_progress=False,
            execution_progress=False, post_mutation_read_allowed=False):
        if name in self.items: raise ValueError(name)
        if (not callable(evidence_policy)
                and evidence_policy not in {"repeatable","read_once","working_memory",
                                            "image_twice","budgeted_output",
                                            "invalidates_reads"}):
            raise ValueError(f"unsupported evidence policy: {evidence_policy}")
        Draft202012Validator.check_schema(dict(parameters))
        self.items[name] = Function(name, description, parameters, call, available,
                                    evidence_policy,str(evidence_group),
                                    tuple(str(item) for item in invalidates_evidence_groups),
                                    evidence_progress,execution_progress,
                                    post_mutation_read_allowed)
    @property
    def schemas(self):
        return [item.schema for item in self.items.values()
                if item.available is None or item.available()]
    def invoke(self, name, arguments):
        if name not in self.items: raise KeyError(name)
        item=self.items[name]
        if item.available is not None and not item.available():
            raise KeyError(f"Tool unavailable in current lifecycle phase: {name}")
        try:Draft202012Validator(dict(item.parameters)).validate(arguments)
        except ValidationError as exc:
            raise ValueError(f"{name} arguments violate schema: {exc.message}") from exc
        return item.call(**dict(arguments))
    def evidence_policy(self, name):
        if name not in self.items: raise KeyError(name)
        return self.items[name].evidence_policy
    def evidence_contract(self, name, arguments=None):
        if name not in self.items: raise KeyError(name)
        item=self.items[name]
        progress=(item.evidence_progress(dict(arguments or {}))
                  if callable(item.evidence_progress) else bool(item.evidence_progress))
        execution=(item.execution_progress(dict(arguments or {}))
                   if callable(item.execution_progress) else bool(item.execution_progress))
        post_mutation_allowed=(item.post_mutation_read_allowed(dict(arguments or {}))
                    if callable(item.post_mutation_read_allowed)
                    else bool(item.post_mutation_read_allowed))
        policy=(item.evidence_policy(dict(arguments or {}))
                if callable(item.evidence_policy) else item.evidence_policy)
        if policy not in {"repeatable","read_once","working_memory","image_twice",
                          "budgeted_output","invalidates_reads"}:
            raise ValueError(f"unsupported dynamic evidence policy: {policy}")
        return {"policy":policy,"group":item.evidence_group,
                "invalidates":item.invalidates_evidence_groups,
                "progress":progress,"execution_progress":execution,
                "post_mutation_read_allowed":post_mutation_allowed}

__all__ = ["FunctionRegistry"]
