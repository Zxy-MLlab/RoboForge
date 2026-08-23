"""Audited internet discovery tools for the Embodied Harness."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote_plus
from urllib.request import Request, urlopen


def _fetch_json(url: str, *, timeout: float = 15.0) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "embodied-frontier-harness/0.1",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type or body.lstrip().startswith(("{", "[")):
            return json.loads(body)
        return body


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def search_public_resources(
    query: str,
    *,
    ledger_path: str = "artifacts/capability_acquisition.jsonl",
    limit: int = 5,
    fetch_json: Callable[[str], Any] = _fetch_json,
) -> dict[str, Any]:
    """Search the open web plus code/model indexes for untrusted leads.

    Search results are untrusted leads. The returned records deliberately omit
    evaluator state and are written to an append-only acquisition ledger.
    """
    query = str(query).strip()
    if not query:
        return {"success": False, "reason": "query must not be empty"}
    if not 1 <= int(limit) <= 20:
        return {"success": False, "reason": "limit must be between 1 and 20"}

    stopwords = {"a", "an", "and", "for", "how", "to", "the", "with", "on", "in", "of"}
    salient = [token for token in re.findall(r"[a-z0-9-]+", query.lower()) if token not in stopwords]
    fallback_query = " ".join(salient[-3:]) if len(salient) >= 3 else " ".join(salient)
    domain_queries: list[str] = []
    lowered = query.casefold()
    if any(token in lowered for token in ("visual servo", "visual servoing", "servoing", "button", "press")):
        domain_queries.extend(
            [
                "visual servo robot",
                "robot button pressing",
                "robot manipulation skill library",
            ]
        )
    if any(token in lowered for token in ("grasp", "gripper", "slip", "friction")):
        domain_queries.append("robot grasp policy")
    if "collision" in lowered and any(token in lowered for token in ("support", "surface", "stack")):
        domain_queries.insert(0, "collision aware grasp generation support surface")
    if any(token in lowered for token in ("rgbd", "depth", "table", "plane", "camera")):
        domain_queries.append("robot rgbd perception")

    # Preserve the model's specific hypothesis first. Broad domain expansions
    # are fallbacks and are all searched; a generic early GitHub hit must not
    # suppress more relevant Contact-GraspNet / collision-filtering results.
    queries = [query, *domain_queries]
    if fallback_query and fallback_query != query:
        queries.append(fallback_query)
    queries = list(dict.fromkeys(queries))
    endpoints = {}
    results: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    attempted_queries: list[str] = []
    for candidate_query in queries:
        attempted_queries.append(candidate_query)
        endpoints[candidate_query] = {
            "github": (
                "https://api.github.com/search/repositories?q="
                + quote_plus(candidate_query)
                + f"&per_page={int(limit)}"
            ),
            "huggingface": (
                "https://huggingface.co/api/models?search="
                + quote_plus(candidate_query)
                + f"&limit={int(limit)}"
            ),
            "web": (
                "https://html.duckduckgo.com/html/?q="
                + quote_plus(candidate_query)
            ),
            "arxiv": (
                "https://export.arxiv.org/api/query?search_query=all:"
                + quote_plus(candidate_query)
                + f"&max_results={int(limit)}"
            ),
            "crossref": (
                "https://api.crossref.org/works?query="
                + quote_plus(candidate_query)
                + f"&rows={int(limit)}"
            ),
        }
        for source, endpoint in endpoints[candidate_query].items():
            try:
                payload = fetch_json(endpoint)
                if source == "github":
                    for item in list(payload.get("items", []))[:limit]:
                        results.append(
                            {
                                "source": source,
                                "name": item.get("full_name"),
                                "url": item.get("html_url"),
                                "revision": item.get("default_branch"),
                                "description": item.get("description"),
                                "stars": item.get("stargazers_count"),
                            }
                        )
                else:
                    if source == "arxiv":
                        raw = payload if isinstance(payload, str) else str(payload)
                        for match in re.finditer(r"<entry>(.*?)</entry>", raw, re.S):
                            block = match.group(1)
                            ident = re.search(r"<id>(.*?)</id>", block, re.S)
                            title = re.search(r"<title>(.*?)</title>", block, re.S)
                            if ident:
                                results.append({"source": source, "name": (title.group(1).strip() if title else None), "url": ident.group(1).strip(), "description": None})
                        continue
                    if source == "crossref":
                        items = (payload.get("message", {}) if isinstance(payload, dict) else {}).get("items", [])
                        for item in items[:limit]:
                            results.append({"source": source, "name": (item.get("title") or [None])[0], "url": item.get("URL"), "revision": item.get("published-print") or item.get("published-online")})
                        continue
                    if source == "web":
                        raw = payload if isinstance(payload, str) else str(payload)
                        for match in re.finditer(r'(?i)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', raw):
                            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
                            results.append({"source": source, "name": title, "url": match.group(1), "description": None})
                        continue
                    for item in list(payload)[:limit]:
                        model_id = item.get("id")
                        results.append(
                            {
                                "source": source,
                                "name": model_id,
                                "url": f"https://huggingface.co/{model_id}",
                                "revision": item.get("sha"),
                                "pipeline_tag": item.get("pipeline_tag"),
                                "downloads": item.get("downloads"),
                            }
                        )
            except Exception as exc:
                errors[f"{candidate_query}:{source}"] = f"{type(exc).__name__}: {exc}"
        # Search every planned query. Some indexes (especially Crossref) return
        # a superficially non-empty result for nearly any phrase.

    deduplicated: list[dict[str, Any]] = []
    seen_resources: set[tuple[str, str]] = set()
    for item in results:
        identity=(str(item.get("source") or ""),str(item.get("url") or item.get("name") or ""))
        if identity in seen_resources: continue
        seen_resources.add(identity); deduplicated.append(item)
    results=deduplicated

    event = {
        "event": "public_resource_search",
        "timestamp_unix": time.time(),
        "query": query,
        "attempted_queries": attempted_queries,
        "endpoints": endpoints,
        "result_count": len(results),
        "errors": errors,
    }
    _append_event(Path(ledger_path), event)
    if not results and errors:
        return {"success": False, "reason": "all public searches failed", "errors": errors}
    return {
        "success": True,
        "query": query,
        "attempted_queries": attempted_queries,
        "results": results,
        "errors": errors,
    }


def record_acquisition_event(
    event: dict[str, Any],
    *,
    ledger_path: str = "artifacts/capability_acquisition.jsonl",
) -> dict[str, Any]:
    """Persist an explicit install, rejection, registration, or reuse event."""
    if not isinstance(event, dict) or not str(event.get("event", "")).strip():
        return {"success": False, "reason": "event must include a non-empty event field"}
    payload = dict(event)
    payload["timestamp_unix"] = time.time()
    _append_event(Path(ledger_path), payload)
    return {"success": True, "recorded_event": payload.get("event"), "ledger_path": ledger_path}


def register_public_resource_tools(registry: Any) -> None:
    """Register public discovery and acquisition-ledger tools with Thea."""
    registry.tool(
        name="search_public_embodied_resources",
        description=(
            "Search the open web, papers, documentation, GitHub, and Hugging "
            "Face for an embodied failure mode. Results are untrusted leads."
        ),
        # ``fetch_json`` is an internal dependency-injection seam for tests,
        # not a model-facing argument.  An explicit schema prevents its
        # callable default from leaking into OpenAI-compatible Tool JSON.
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "ledger_path": {
                    "type": "string",
                    "default": "artifacts/capability_acquisition.jsonl",
                },
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )(search_public_resources)
    registry.tool(
        name="record_capability_acquisition_event",
        description=(
            "Record a public capability install, rejection, registration, or "
            "reuse event in the append-only acquisition ledger."
        ),
    )(record_acquisition_event)


__all__ = [
    "record_acquisition_event",
    "register_public_resource_tools",
    "search_public_resources",
]
