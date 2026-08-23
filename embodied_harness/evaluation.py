"""Sealed final benchmark evaluation helpers, separate from evolution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapters.libero import LiberoAdapter
from .graph_store import GraphStore


def run_libero_evaluator_once(
    *, graph_id: str, graphs: GraphStore, adapter: LiberoAdapter,
    output_path: str | Path,
) -> dict[str, Any]:
    """Execute a frozen graph, then consume the benchmark evaluator once.

    The evaluator result is written only after execution and is never returned
    to an EvolutionEngine or inserted into controller context.
    """
    execution = graphs.execute(graph_id, adapter)
    sensor_report = dict(adapter.sensor_report(execution))
    benchmark_passed = adapter._sealed_check_once()
    report = {
        "protocol": "sealed-final-evaluation-v1", "graph_id": graph_id,
        "execution_completed": execution.get("completed"),
        "sensor_report": sensor_report, "benchmark_passed": benchmark_passed,
        "evaluator_calls": 1, "fed_back_to_evolution": False,
    }
    destination = Path(output_path).resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    adapter.close()
    return report


__all__ = ["run_libero_evaluator_once"]
