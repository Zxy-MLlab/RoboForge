from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import AdapterResult, RawArtifact


class FakeAdapter:
    def __init__(self) -> None:
        self.execution_kinds: list[str] = []
        self.reset_count = 0
        self.controller_runs = 0
        self.observe_count = 0
        self.public_state: dict[str, Any] = {
            "robot": {"eef_position_m": [0.0, 0.0, 0.0]},
            "verifier": {"facts": {"object_attached": False}},
        }
        self.receipt_verified = False
        self.raise_during_execution: Exception | None = None

    def begin_execution(self, kind: str) -> None:
        self.execution_kinds.append(kind)

    def observe(self) -> AdapterResult:
        self.observe_count += 1
        return AdapterResult(
            public={**self.public_state, "observation_index": self.observe_count},
            artifacts=(
                RawArtifact(
                    name="camera.png",
                    media_type="image/png",
                    data=f"diagnostic-{self.observe_count}".encode("ascii"),
                ),
            ),
        )

    def reset_to_s0(self) -> str:
        self.reset_count += 1
        return f"generation-{self.reset_count}"

    def execute_controller(
        self,
        *,
        controller_path: Path,
        controller_sha256: str,
        environment_generation: str,
    ) -> AdapterResult:
        self.controller_runs += 1
        if self.raise_during_execution is not None:
            raise self.raise_during_execution
        return AdapterResult(
            public={
                **self.public_state,
                "execution": {
                    "controller_run": self.controller_runs,
                    "all_actions_reached": True,
                },
            },
            artifacts=(
                RawArtifact(
                    name="rollout.mp4",
                    media_type="video/mp4",
                    data=f"physical-{self.controller_runs}".encode("ascii"),
                ),
            ),
            private_receipt={
                "kind": "physical",
                "controller_sha256": controller_sha256,
                "environment_generation": environment_generation,
                "verified": self.receipt_verified,
            },
        )

    def validate_receipt(
        self,
        receipt: dict[str, Any],
        *,
        controller_sha256: str,
        environment_generation: str,
    ) -> bool:
        return bool(
            receipt.get("kind") == "physical"
            and receipt.get("controller_sha256") == controller_sha256
            and receipt.get("environment_generation") == environment_generation
            and receipt.get("verified") is True
        )
