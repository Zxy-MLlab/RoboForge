from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..workspace.project import ProjectWorkspace


@dataclass(frozen=True)
class CampaignConfig:
    runtime: str
    task: str
    development_states: tuple[int, ...]
    held_out_split: str
    max_valid_trials: int = 30
    max_iterations: int = 80
    controller_mode: str = "JOINT_POSITION"


def load_campaign_config(path: str | Path) -> CampaignConfig:
    source = Path(path).resolve()
    text = source.read_text(encoding="utf-8")
    try:
        import yaml
        value = yaml.safe_load(text)
    except ImportError:
        value = json.loads(text)
    if not isinstance(value, Mapping):
        raise ValueError("campaign config must be an object")
    states = tuple(sorted(set(int(item) for item in value.get("development_states", ()))))
    if not states:
        raise ValueError("development_states must not be empty")
    return CampaignConfig(
        runtime=str(value.get("runtime", "libero")), task=str(value["task"]),
        development_states=states, held_out_split=str(value["held_out_split"]),
        max_valid_trials=int(value.get("max_valid_trials", 30)),
        max_iterations=int(value.get("max_iterations", 80)),
        controller_mode=str(value.get("controller_mode", "JOINT_POSITION")),
    )


@dataclass
class CanonicalCampaign:
    config: CampaignConfig
    workspace: ProjectWorkspace
    output: Path
    trial_runner: Callable[[int, Path], Mapping[str, Any]] | None = None
    records: list[dict[str, Any]] = field(default_factory=list)

    def run(self) -> dict[str, Any]:
        """Run development states against one shared editable workspace.

        The runner is deliberately injected by the CLI/provider. This class
        owns scheduling and accounting, never coding-model decisions.
        """
        self.workspace.initialize()
        self.output.mkdir(parents=True, exist_ok=True)
        started = time.time()
        for state in self.config.development_states:
            if len([r for r in self.records if r.get("valid_trial")]) >= self.config.max_valid_trials:
                break
            if self.trial_runner is None:
                break
            result = dict(self.trial_runner(state, self.workspace.root))
            result.setdefault("state", state)
            self.records.append(result)
        payload = {
            "schema_version": "roboforge-canonical-campaign-v1",
            "runtime": self.config.runtime, "task": self.config.task,
            "development_states": list(self.config.development_states),
            "max_valid_trials": self.config.max_valid_trials,
            "records": self.records,
            "termination_reason": "valid_trial_budget_exhausted" if len(self.records) >= self.config.max_valid_trials else "runner_complete",
            "elapsed_seconds": time.time() - started,
        }
        (self.output / "campaign.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
