"""Structured immutable controller graphs composed from typed Stage Nodes."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

from stage_node_workspace import (
    StageNodeValidationError,
    StageNodeWorkspace,
    audit_stage_return_contract,
)


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_ALIAS = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TERMINALS = {"$success", "$failure"}


class ControllerGraphValidationError(ValueError):
    pass


class ControllerGraphWorkspace:
    """Graph manifests freeze node IDs and topology, never source substrings."""

    def __init__(
        self, root: str | Path, *, nodes: StageNodeWorkspace,
        required_revision: Mapping[str, Any] | None = None,
        require_checkpoint_success: bool = False,
        available_initial_context_fields: set[str] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.nodes = nodes
        self.required_revision = dict(required_revision or {})
        self.require_checkpoint_success = bool(require_checkpoint_success)
        self.available_initial_context_fields = (
            None if available_initial_context_fields is None
            else {str(field) for field in available_initial_context_fields}
        )

    def _validate_spec(
        self,
        *,
        entry_node: str,
        bindings: Mapping[str, str],
        edges: list[Mapping[str, str]],
        initial_context_fields: list[str],
        max_node_visits: int,
    ) -> dict[str, Any]:
        normalized_bindings = {str(alias): str(node_id) for alias, node_id in bindings.items()}
        if not normalized_bindings or any(
            not _ALIAS.fullmatch(alias) for alias in normalized_bindings
        ):
            raise ControllerGraphValidationError("graph bindings require valid node aliases")
        if entry_node not in normalized_bindings:
            raise ControllerGraphValidationError("entry_node is not bound")
        if not 1 <= int(max_node_visits) <= 20:
            raise ControllerGraphValidationError("max_node_visits must be within [1, 20]")
        manifests = {
            alias: self.nodes.inspect(node_id)["manifest"]
            for alias, node_id in normalized_bindings.items()
        }
        normalized_edges = []
        routes: dict[tuple[str, str], str] = {}
        for raw in edges:
            source = str(raw.get("from") or "")
            outcome = str(raw.get("outcome") or "")
            target = str(raw.get("to") or "")
            if source not in normalized_bindings:
                raise ControllerGraphValidationError(f"edge source is not bound: {source}")
            if outcome not in manifests[source]["allowed_outcomes"]:
                raise ControllerGraphValidationError(
                    f"edge uses undeclared outcome: {source}/{outcome}"
                )
            if target not in normalized_bindings and target not in _TERMINALS:
                raise ControllerGraphValidationError(f"edge target is invalid: {target}")
            key = (source, outcome)
            if key in routes:
                raise ControllerGraphValidationError(f"duplicate graph route: {source}/{outcome}")
            routes[key] = target
            normalized_edges.append({"from": source, "outcome": outcome, "to": target})
        for alias, manifest in manifests.items():
            missing = set(manifest["allowed_outcomes"]) - {
                outcome for source, outcome in routes if source == alias
            }
            if missing:
                raise ControllerGraphValidationError(
                    f"node outcomes have no routes: {alias}/{sorted(missing)}"
                )

        reachable = {entry_node}
        queue = deque([entry_node])
        terminals = set()
        while queue:
            source = queue.popleft()
            for (edge_source, _), target in routes.items():
                if edge_source != source:
                    continue
                if target in _TERMINALS:
                    terminals.add(target)
                elif target not in reachable:
                    reachable.add(target)
                    queue.append(target)
        unreachable = set(normalized_bindings) - reachable
        if unreachable:
            raise ControllerGraphValidationError(f"unreachable graph nodes: {sorted(unreachable)}")
        if not terminals:
            raise ControllerGraphValidationError("graph has no reachable terminal")

        if self.require_checkpoint_success:
            proof_queue = deque([(entry_node, False)])
            proof_visited = set()
            while proof_queue:
                source, verified = proof_queue.popleft()
                state = (source, verified)
                if state in proof_visited:
                    continue
                proof_visited.add(state)
                for (edge_source, outcome), target in routes.items():
                    if edge_source != source:
                        continue
                    next_verified = verified or outcome in set(
                        manifests[source].get("checkpoint_outcomes") or []
                    )
                    if target == "$success" and not next_verified:
                        raise ControllerGraphValidationError(
                            "every $success path requires an adapter-verified checkpoint"
                        )
                    if target not in _TERMINALS:
                        proof_queue.append((target, next_verified))

        initial = list(dict.fromkeys(str(field) for field in initial_context_fields))
        if any(not _ALIAS.fullmatch(field) for field in initial):
            raise ControllerGraphValidationError("invalid initial context field")
        if self.available_initial_context_fields is not None:
            unsupported = set(initial) - self.available_initial_context_fields
            if unsupported:
                raise ControllerGraphValidationError(
                    "initial context fields are not supplied by this Robot Adapter: "
                    f"{sorted(unsupported)}; available="
                    f"{sorted(self.available_initial_context_fields)}"
                )
        available: dict[str, set[str]] = {entry_node: set(initial)}
        changed = True
        while changed:
            changed = False
            incoming: dict[str, list[set[str]]] = defaultdict(list)
            for (source, outcome), target in routes.items():
                if source not in available or target in _TERMINALS or target == entry_node:
                    continue
                incoming[target].append(
                    available[source]
                    | set(manifests[source]["provides_by_outcome"].get(outcome) or ())
                )
            for target, paths in incoming.items():
                guaranteed = set.intersection(*paths) if paths else set()
                if target not in available or available[target] != guaranteed:
                    available[target] = guaranteed
                    changed = True
        for alias, manifest in manifests.items():
            missing = set(manifest["requires"]) - available.get(alias, set())
            if missing:
                raise ControllerGraphValidationError(
                    f"node requirements are not guaranteed on every path: "
                    f"{alias}/{sorted(missing)}"
                )
        return {
            "entry_node": entry_node,
            "bindings": normalized_bindings,
            "edges": normalized_edges,
            "initial_context_fields": initial,
            "max_node_visits": int(max_node_visits),
            "reachable_terminals": sorted(terminals),
        }

    def create(
        self,
        *,
        name: str,
        description: str,
        entry_node: str,
        bindings: Mapping[str, str],
        edges: list[Mapping[str, str]],
        initial_context_fields: list[str] | None = None,
        max_node_visits: int = 3,
        base_graph_id: str | None = None,
        frozen_node_aliases: list[str] | None = None,
    ) -> dict[str, Any]:
        if not _NAME.fullmatch(str(name)):
            raise ControllerGraphValidationError("invalid graph name")
        spec = self._validate_spec(
            entry_node=str(entry_node), bindings=bindings, edges=edges,
            initial_context_fields=list(initial_context_fields or []),
            max_node_visits=max_node_visits,
        )
        frozen = list(dict.fromkeys(str(alias) for alias in (frozen_node_aliases or [])))
        if self.required_revision:
            required_base = str(self.required_revision.get("base_graph_id") or "")
            required_frozen = list(self.required_revision.get("frozen_node_aliases") or [])
            if str(base_graph_id or "") != required_base:
                raise ControllerGraphValidationError(
                    f"required_base_graph:{required_base}"
                )
            if set(frozen) != set(required_frozen):
                raise ControllerGraphValidationError(
                    f"required_frozen_node_aliases:{sorted(required_frozen)}"
                )
        revision = None
        if base_graph_id is not None:
            base = self.inspect(base_graph_id)["manifest"]
            unknown = set(frozen) - set(base["bindings"])
            if unknown:
                raise ControllerGraphValidationError(
                    f"frozen aliases are absent from base graph: {sorted(unknown)}"
                )
            for alias in frozen:
                if spec["bindings"].get(alias) != base["bindings"][alias]:
                    raise ControllerGraphValidationError(
                        f"frozen_node_replaced:{alias}:{base['bindings'][alias]}"
                    )
            base_routes = {
                (edge["from"], edge["outcome"], edge["to"])
                for edge in base["edges"]
                if edge["from"] in frozen or edge["to"] in frozen
            }
            new_routes = {
                (edge["from"], edge["outcome"], edge["to"])
                for edge in spec["edges"]
                if edge["from"] in frozen or edge["to"] in frozen
            }
            if new_routes != base_routes:
                raise ControllerGraphValidationError("frozen_node_topology_changed")
            if base["entry_node"] in frozen and spec["entry_node"] != base["entry_node"]:
                raise ControllerGraphValidationError("frozen_entry_node_changed")
            revision = {
                "base_graph_id": base_graph_id,
                "frozen_node_aliases": frozen,
                "frozen_node_ids": {alias: base["bindings"][alias] for alias in frozen},
            }
        elif frozen:
            raise ControllerGraphValidationError("frozen aliases require base_graph_id")

        family = self.root / name
        versions = [
            int(path.name[1:]) for path in family.glob("v[0-9]*")
            if path.name[1:].isdigit()
        ]
        version = max(versions, default=0) + 1
        destination = family / f"v{version:03d}"
        destination.mkdir(parents=True, exist_ok=False)
        manifest = {
            "protocol": "embodied-controller-graph-v1",
            "name": name, "version": version,
            "graph_id": f"{name}:v{version:03d}",
            "description": str(description),
            **spec,
            "revision": revision,
            "status": "candidate",
            "created_unix": time.time(),
            "privileged_state_used": False,
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["graph_sha256"] = hashlib.sha256(canonical).hexdigest()
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return {
            "success": True, "graph_id": manifest["graph_id"],
            "graph_sha256": manifest["graph_sha256"], "revision": revision,
        }

    def resolve(self, graph_id: str) -> Path:
        name, separator, version = str(graph_id).partition(":")
        if not separator or not _NAME.fullmatch(name) or not re.fullmatch(r"v[0-9]{3}", version):
            raise ControllerGraphValidationError("invalid graph_id")
        destination = (self.root / name / version).resolve()
        if self.root not in destination.parents or not (destination / "manifest.json").is_file():
            raise FileNotFoundError(graph_id)
        return destination

    def inspect(self, graph_id: str) -> dict[str, Any]:
        destination = self.resolve(graph_id)
        manifest = json.loads((destination / "manifest.json").read_text())
        expected = manifest.pop("graph_sha256")
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["graph_sha256"] = expected
        if hashlib.sha256(canonical).hexdigest() != expected:
            raise ControllerGraphValidationError("immutable Controller Graph hash mismatch")
        for node_id in manifest["bindings"].values():
            self.nodes.inspect(node_id)
        return {"success": True, "graph_id": graph_id, "manifest": manifest}

    def preflight(self, graph_id: str) -> dict[str, Any]:
        """Compile graph and node contracts without launching a Robot Adapter."""
        manifest = self.inspect(graph_id)["manifest"]
        self._validate_spec(
            entry_node=manifest["entry_node"], bindings=manifest["bindings"],
            edges=manifest["edges"],
            initial_context_fields=manifest.get("initial_context_fields") or [],
            max_node_visits=int(manifest["max_node_visits"]),
        )
        compiled_nodes = []
        for alias, node_id in manifest["bindings"].items():
            inspected = self.nodes.inspect(node_id)
            node_manifest = inspected["manifest"]
            contract = audit_stage_return_contract(
                inspected["source"], node_manifest["provides_by_outcome"]
            )
            if not contract["eligible"]:
                raise ControllerGraphValidationError(
                    f"Stage Node preflight failed: {alias}/{node_id}: "
                    f"{contract['violations']}"
                )
            compiled_nodes.append({
                "alias": alias, "node_id": node_id,
                "sha256": node_manifest["sha256"],
                "implemented_outcomes": contract["implemented_outcomes"],
                "checkpoint_outcomes": node_manifest.get("checkpoint_outcomes") or [],
            })
        return {
            "success": True, "eligible": True, "graph_id": graph_id,
            "graph_sha256": manifest["graph_sha256"],
            "node_count": len(compiled_nodes), "nodes": compiled_nodes,
            "adapter_launched": False, "evaluator_used": False,
        }

    def execute(
        self,
        graph_id: str,
        dispatch: Callable[[str, Mapping[str, Any]], Any],
        initial_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.preflight(graph_id)
        manifest = self.inspect(graph_id)["manifest"]
        context = dict(initial_context or {})
        missing_initial = set(manifest["initial_context_fields"]) - set(context)
        if missing_initial:
            raise ControllerGraphValidationError(
                f"missing initial graph context: {sorted(missing_initial)}"
            )
        routes = {
            (edge["from"], edge["outcome"]): edge["to"]
            for edge in manifest["edges"]
        }
        current = manifest["entry_node"]
        visits: Counter[str] = Counter()
        node_trace = []
        rpc_events = []
        executed_aliases = []
        verified_prefix_aliases = []
        while current not in _TERMINALS:
            visits[current] += 1
            if visits[current] > manifest["max_node_visits"]:
                return {
                    "execution_completed": False,
                    "error": f"Stage Node visit budget exceeded: {current}",
                    "graph_id": graph_id, "node_trace": node_trace,
                    "rpc_events": rpc_events,
                    "verified_prefix_aliases": verified_prefix_aliases,
                }
            node_id = manifest["bindings"][current]
            node_manifest = self.nodes.inspect(node_id)["manifest"]
            try:
                report = self.nodes.execute(node_id, context, dispatch)
            except StageNodeValidationError as exc:
                node_trace.append({
                    "alias": current, "node_id": node_id,
                    "execution_completed": False, "outcome": None,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                return {
                    "execution_completed": False,
                    "error": f"Stage Node contract failed: {current}: {exc}",
                    "graph_id": graph_id, "node_trace": node_trace,
                    "rpc_events": rpc_events,
                    "verified_prefix_aliases": verified_prefix_aliases,
                }
            rpc_events.extend(report.get("rpc_events") or [])
            trace_row = {
                "alias": current, "node_id": node_id,
                "execution_completed": bool(report.get("execution_completed")),
                "outcome": report.get("outcome"), "error": report.get("error"),
            }
            node_trace.append(trace_row)
            executed_aliases.append(current)
            if not report.get("execution_completed"):
                return {
                    "execution_completed": False,
                    "error": report.get("error") or f"Stage Node failed: {current}",
                    "graph_id": graph_id, "node_trace": node_trace,
                    "rpc_events": rpc_events,
                    "verified_prefix_aliases": verified_prefix_aliases,
                }
            context.update(report["updates"])
            if report["outcome"] in node_manifest.get("checkpoint_outcomes", []):
                verifier_events = [
                    event for event in (report.get("rpc_events") or [])
                    if event.get("method") == "call_tool"
                    and str(((event.get("arguments") or {}).get("name") or "")).startswith(
                        "verify_"
                    )
                ]
                verifier_result = (
                    (verifier_events[-1].get("result") or {})
                    if verifier_events else {}
                )
                if verifier_result.get("verified") is not True:
                    trace_row["execution_completed"] = False
                    trace_row["error"] = (
                        "checkpoint outcome lacks adapter-owned verified:true evidence"
                    )
                    return {
                        "execution_completed": False,
                        "error": trace_row["error"],
                        "graph_id": graph_id,
                        "node_trace": node_trace,
                        "rpc_events": rpc_events,
                        "verified_prefix_aliases": verified_prefix_aliases,
                    }
                verified_prefix_aliases = list(dict.fromkeys(executed_aliases))
            current = routes[(current, report["outcome"])]
        return {
            "execution_completed": True,
            "result": {"graph_outcome": current[1:], "context": context},
            "graph_outcome": current[1:], "graph_id": graph_id,
            "node_trace": node_trace, "rpc_events": rpc_events,
            "verified_prefix_aliases": verified_prefix_aliases,
        }


def register_controller_graph_tools(
    registry: Any,
    workspace: ControllerGraphWorkspace,
    *,
    executor: Callable[[str], Mapping[str, Any]] | None = None,
) -> None:
    @registry.tool(
        name="create_controller_graph",
        description=(
            "Create an immutable structured controller graph by binding Stage Node IDs. "
            "Route terminal outcomes directly to the literal targets $success or "
            "$failure; do not create terminal Stage Nodes. "
            "For revisions, frozen_node_aliases must retain exactly the base node IDs "
            "and incident edges; replace only failed nodes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 3, "maxLength": 80},
                "description": {"type": "string", "maxLength": 3000},
                "entry_node": {"type": "string"},
                "bindings": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "edges": {
                    "type": "array", "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string"},
                            "outcome": {"type": "string"},
                            "to": {
                                "type": "string",
                                "description": (
                                    "Bound node alias or exact literal $success/$failure."
                                ),
                            },
                        },
                        "required": ["from", "outcome", "to"],
                        "additionalProperties": False,
                    },
                },
                "initial_context_fields": {
                    "type": "array", "items": {"type": "string"},
                },
                "max_node_visits": {"type": "integer", "minimum": 1, "maximum": 20},
                "base_graph_id": {"type": ["string", "null"]},
                "frozen_node_aliases": {
                    "type": "array", "items": {"type": "string"},
                },
            },
            "required": ["name", "description", "entry_node", "bindings", "edges"],
            "additionalProperties": False,
        },
    )
    def create_controller_graph(
        name: str, description: str, entry_node: str,
        bindings: Mapping[str, str], edges: list[Mapping[str, str]],
        initial_context_fields: list[str] | None = None,
        max_node_visits: int = 3, base_graph_id: str | None = None,
        frozen_node_aliases: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            return workspace.create(
                name=name, description=description, entry_node=entry_node,
                bindings=bindings, edges=edges,
                initial_context_fields=initial_context_fields,
                max_node_visits=max_node_visits, base_graph_id=base_graph_id,
                frozen_node_aliases=frozen_node_aliases,
            )
        except (ControllerGraphValidationError, FileNotFoundError) as exc:
            return {"success": False, "graph_created": False, "reason": str(exc)}

    @registry.tool(
        name="inspect_controller_graph",
        description="Inspect a graph's node bindings, topology, revision freeze, and hash.",
        input_schema={
            "type": "object",
            "properties": {"graph_id": {"type": "string"}},
            "required": ["graph_id"], "additionalProperties": False,
        },
    )
    def inspect_controller_graph(graph_id: str) -> dict[str, Any]:
        try:
            return workspace.inspect(graph_id)
        except (ControllerGraphValidationError, FileNotFoundError) as exc:
            return {"success": False, "reason": str(exc)}

    @registry.tool(
        name="preflight_controller_graph",
        description=(
            "Compile graph topology, context paths, terminals, hashes, checkpoints, "
            "and every Stage Node return contract without launching a Robot Adapter."
        ),
        input_schema={
            "type": "object",
            "properties": {"graph_id": {"type": "string"}},
            "required": ["graph_id"], "additionalProperties": False,
        },
    )
    def preflight_controller_graph(graph_id: str) -> dict[str, Any]:
        try:
            return workspace.preflight(graph_id)
        except (ControllerGraphValidationError, StageNodeValidationError,
                FileNotFoundError) as exc:
            return {
                "success": False, "eligible": False,
                "adapter_launched": False, "reason": str(exc),
            }

    if executor is not None:
        @registry.tool(
            name="execute_controller_graph",
            description=(
                "Execute one immutable Controller Graph on the deployment-owned "
                "persistent Robot Adapter and return sensor-only evidence."
            ),
            input_schema={
                "type": "object",
                "properties": {"graph_id": {"type": "string"}},
                "required": ["graph_id"], "additionalProperties": False,
            },
        )
        def execute_controller_graph(graph_id: str) -> dict[str, Any]:
            preflight = preflight_controller_graph(graph_id)
            if not preflight.get("success"):
                return preflight
            result = dict(executor(graph_id))
            result.setdefault("success", True)
            return result


__all__ = [
    "ControllerGraphValidationError", "ControllerGraphWorkspace",
    "register_controller_graph_tools",
]
