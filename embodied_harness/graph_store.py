"""Immutable Controller Graph compiler and persistent-adapter executor."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

from .adapter import RobotAdapter
from .errors import GraphCompileError, NodeCompileError, NodeRuntimeError
from .node_store import NodeStore


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_ALIAS = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TERMINALS = {"$success", "$failure"}


class GraphStore:
    def __init__(
        self, root: str | Path, *, nodes: NodeStore,
        available_initial_fields: set[str], require_verified_success: bool = True,
        required_revision: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.nodes = nodes
        self.available_initial_fields = set(available_initial_fields)
        self.require_verified_success = bool(require_verified_success)
        self.required_revision = dict(required_revision or {})

    def _compile_spec(
        self, *, entry: str, bindings: Mapping[str, str],
        edges: list[Mapping[str, str]], initial_fields: list[str],
        max_visits: int,
    ) -> dict[str, Any]:
        bound = {str(alias): str(node_id) for alias, node_id in bindings.items()}
        if not bound or entry not in bound or any(not _ALIAS.fullmatch(a) for a in bound):
            raise GraphCompileError("invalid entry or node bindings")
        if not 1 <= int(max_visits) <= 20: raise GraphCompileError("invalid visit budget")
        manifests = {alias: self.nodes.inspect(node_id)["manifest"]
                     for alias, node_id in bound.items()}
        initial = list(dict.fromkeys(map(str, initial_fields)))
        unsupported = set(initial) - self.available_initial_fields
        if unsupported:
            raise GraphCompileError(
                f"adapter does not supply initial fields: {sorted(unsupported)}"
            )
        routes: dict[tuple[str, str], str] = {}; normalized_edges = []
        for raw in edges:
            source, outcome, target = map(str, (raw.get("from", ""),
                                               raw.get("outcome", ""), raw.get("to", "")))
            if source not in bound: raise GraphCompileError(f"unbound edge source: {source}")
            if outcome not in manifests[source]["outcomes"]:
                raise GraphCompileError(f"undeclared edge outcome: {source}/{outcome}")
            if target not in bound and target not in _TERMINALS:
                raise GraphCompileError(f"invalid edge target: {target}")
            if (source, outcome) in routes:
                raise GraphCompileError(f"duplicate route: {source}/{outcome}")
            routes[(source, outcome)] = target
            normalized_edges.append({"from": source, "outcome": outcome, "to": target})
        for alias, manifest in manifests.items():
            missing = set(manifest["outcomes"]) - {
                outcome for source, outcome in routes if source == alias
            }
            if missing: raise GraphCompileError(f"missing routes: {alias}/{sorted(missing)}")

        reachable = {entry}; queue = deque([entry]); terminals = set()
        while queue:
            source = queue.popleft()
            for (edge_source, _), target in routes.items():
                if edge_source != source: continue
                if target in _TERMINALS: terminals.add(target)
                elif target not in reachable: reachable.add(target); queue.append(target)
        if set(bound) - reachable:
            raise GraphCompileError(f"unreachable nodes: {sorted(set(bound)-reachable)}")
        if not terminals: raise GraphCompileError("graph has no terminal")

        guaranteed: dict[str, set[str]] = {entry: set(initial)}
        changed = True
        while changed:
            changed = False; incoming: dict[str, list[set[str]]] = defaultdict(list)
            for (source, outcome), target in routes.items():
                if source not in guaranteed or target in _TERMINALS or target == entry: continue
                incoming[target].append(
                    guaranteed[source] | set(manifests[source]["provides_by_outcome"][outcome])
                )
            for target, paths in incoming.items():
                fields = set.intersection(*paths) if paths else set()
                if guaranteed.get(target) != fields: guaranteed[target] = fields; changed = True
        for alias, manifest in manifests.items():
            missing = set(manifest["requires"]) - guaranteed.get(alias, set())
            if missing: raise GraphCompileError(
                f"inputs not guaranteed on every path: {alias}/{sorted(missing)}"
            )

        if self.require_verified_success:
            # A previous observation checkpoint does not prove that a later
            # manipulation succeeded.  The edge entering $success must itself
            # be a sensor-verified checkpoint outcome.
            for (source, outcome), target in routes.items():
                if target == "$success" and outcome not in set(
                    manifests[source]["checkpoint_outcomes"]
                ):
                    raise GraphCompileError(
                        "$success must be entered directly from a verified checkpoint"
                    )
        return {
            "entry": entry, "bindings": bound, "edges": normalized_edges,
            "initial_fields": initial, "max_visits": int(max_visits),
            "reachable_terminals": sorted(terminals),
        }

    def create(
        self, *, name: str, description: str, entry: str,
        bindings: Mapping[str, str], edges: list[Mapping[str, str]],
        initial_fields: list[str] | None = None, max_visits: int = 3,
        base_graph_id: str | None = None, frozen_aliases: list[str] | None = None,
    ) -> dict[str, Any]:
        if not _NAME.fullmatch(name): raise GraphCompileError("invalid graph name")
        compiled = self._compile_spec(
            entry=entry, bindings=bindings, edges=edges,
            initial_fields=list(initial_fields or []), max_visits=max_visits,
        )
        frozen = list(dict.fromkeys(map(str, frozen_aliases or [])))
        if self.required_revision:
            if base_graph_id != self.required_revision.get("base_graph_id"):
                raise GraphCompileError("required base graph was not used")
            if set(frozen) != set(self.required_revision.get("frozen_aliases") or []):
                raise GraphCompileError("required frozen aliases differ")
        revision = None
        if base_graph_id:
            base = self.inspect(base_graph_id)["manifest"]
            for alias in frozen:
                if compiled["bindings"].get(alias) != base["bindings"].get(alias):
                    raise GraphCompileError(f"frozen node replaced: {alias}")
            base_internal = {(e["from"], e["outcome"], e["to"]) for e in base["edges"]
                             if e["from"] in frozen and e["to"] in frozen}
            new_internal = {(e["from"], e["outcome"], e["to"]) for e in compiled["edges"]
                            if e["from"] in frozen and e["to"] in frozen}
            if base_internal != new_internal:
                raise GraphCompileError("frozen subgraph topology changed")
            revision = {"base_graph_id": base_graph_id, "frozen_aliases": frozen,
                        "frozen_node_ids": {a: base["bindings"][a] for a in frozen}}
        elif frozen: raise GraphCompileError("frozen aliases require base graph")

        family = self.root / name
        versions = [int(p.name[1:]) for p in family.glob("v[0-9]*") if p.name[1:].isdigit()]
        version = max(versions, default=0) + 1
        destination = family / f"v{version:03d}"; destination.mkdir(parents=True)
        manifest = {
            "protocol": "standalone-embodied-controller-graph-v1",
            "graph_id": f"{name}:v{version:03d}", "name": name,
            "version": version, "description": description, **compiled,
            "revision": revision, "created_unix": time.time(),
            "privileged_state_used": False,
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["graph_sha256"] = hashlib.sha256(canonical).hexdigest()
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        return {"graph_id": manifest["graph_id"], "graph_sha256": manifest["graph_sha256"],
                "revision": revision}

    def resolve(self, graph_id: str) -> Path:
        name, sep, version = graph_id.partition(":")
        if not sep or not _NAME.fullmatch(name) or not re.fullmatch(r"v[0-9]{3}", version):
            raise GraphCompileError("invalid graph_id")
        path = (self.root / name / version).resolve()
        if self.root not in path.parents or not (path / "manifest.json").is_file():
            raise FileNotFoundError(graph_id)
        return path

    def inspect(self, graph_id: str) -> dict[str, Any]:
        path = self.resolve(graph_id); manifest = json.loads((path/"manifest.json").read_text())
        digest = manifest.pop("graph_sha256")
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["graph_sha256"] = digest
        if hashlib.sha256(canonical).hexdigest() != digest:
            raise GraphCompileError("immutable graph hash mismatch")
        return {"manifest": manifest, "path": str(path)}

    def compile(self, graph_id: str) -> dict[str, Any]:
        manifest = self.inspect(graph_id)["manifest"]
        self._compile_spec(
            entry=manifest["entry"], bindings=manifest["bindings"],
            edges=manifest["edges"], initial_fields=manifest["initial_fields"],
            max_visits=manifest["max_visits"],
        )
        return {"eligible": True, "graph_id": graph_id,
                "graph_sha256": manifest["graph_sha256"],
                "nodes": [{"alias": a, "node_id": n,
                           "source_sha256": self.nodes.inspect(n)["manifest"]["source_sha256"]}
                          for a, n in manifest["bindings"].items()],
                "adapter_launched": False}

    def execute(self, graph_id: str, adapter: RobotAdapter) -> dict[str, Any]:
        self.compile(graph_id); manifest = self.inspect(graph_id)["manifest"]
        context = dict(adapter.initial_context)
        missing = set(manifest["initial_fields"]) - set(context)
        if missing: raise GraphCompileError(f"adapter omitted initial fields: {sorted(missing)}")
        routes = {(e["from"], e["outcome"]): e["to"] for e in manifest["edges"]}
        current = manifest["entry"]; visits: Counter[str] = Counter()
        trace = []; rpc_events = []; executed = []; verified_prefix = []
        while current not in _TERMINALS:
            visits[current] += 1
            if visits[current] > manifest["max_visits"]:
                return {"completed": False, "error": f"visit budget exceeded: {current}",
                        "graph_id": graph_id, "node_trace": trace,
                        "rpc_events": rpc_events, "verified_prefix_aliases": verified_prefix}
            node_id = manifest["bindings"][current]
            try: report = self.nodes.execute(node_id, context, adapter.dispatch)
            except (NodeCompileError, NodeRuntimeError) as exc:
                trace.append({"alias": current, "node_id": node_id,
                              "completed": False, "error": str(exc)})
                return {"completed": False, "error": str(exc), "graph_id": graph_id,
                        "node_trace": trace, "rpc_events": rpc_events,
                        "verified_prefix_aliases": verified_prefix}
            rpc_events.extend(report.get("rpc_events") or [])
            row = {"alias": current, "node_id": node_id,
                   "completed": bool(report.get("completed")),
                   "outcome": report.get("outcome"), "error": report.get("error")}
            trace.append(row); executed.append(current)
            if not report.get("completed"):
                return {"completed": False, "error": report.get("error"),
                        "graph_id": graph_id, "node_trace": trace,
                        "rpc_events": rpc_events, "verified_prefix_aliases": verified_prefix}
            context.update(report["updates"])
            node_manifest = self.nodes.inspect(node_id)["manifest"]
            if report["outcome"] in node_manifest["checkpoint_outcomes"]:
                verifications = [event for event in report["rpc_events"]
                                 if event["method"] == "verify"]
                if not verifications or (verifications[-1].get("result") or {}).get("verified") is not True:
                    row["completed"] = False; row["error"] = "checkpoint lacks verified receipt"
                    return {"completed": False, "error": row["error"], "graph_id": graph_id,
                            "node_trace": trace, "rpc_events": rpc_events,
                            "verified_prefix_aliases": verified_prefix}
                verified_prefix = list(dict.fromkeys(executed))
            current = routes[(current, report["outcome"])]
        return {"completed": True, "graph_id": graph_id,
                "graph_outcome": current[1:], "node_trace": trace,
                "rpc_events": rpc_events, "context": context,
                "verified_prefix_aliases": verified_prefix}


__all__ = ["GraphStore"]
