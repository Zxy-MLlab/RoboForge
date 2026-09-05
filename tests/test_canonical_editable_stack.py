from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embodied_codex.adapters.franka_libero_api import FrankaLiberoApi
from roboforge.harness.split import create_split_manifest
from roboforge.workspace.project import ProjectWorkspace
from roboforge.providers.libero.provider import LiberoProvider


class _Env:
    def get_observation(self):
        return {"robot_joint_pos": np.zeros(7)}


def test_workspace_bootstraps_editable_robot_stack(tmp_path):
    entrypoint = ProjectWorkspace(tmp_path).initialize()
    assert entrypoint == tmp_path / "controllers/controller.py"
    assert (tmp_path / "robot_sdk/franka_libero_api.py").is_file()
    assert (tmp_path / "robot_sdk/libero_sdk.py").is_file()
    assert json.loads((tmp_path / "robot_sdk/BOOTSTRAP_SOURCES.json").read_text())["aspire"]


def test_split_manifest_is_immutable(tmp_path):
    path = tmp_path / "split.json"
    first = create_split_manifest(path, task="4", development=[1, 2], contaminated=[0], final_held_out=[3, 4])
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError, match="immutable"):
        create_split_manifest(path, task="4", development=[1, 2], contaminated=[0], final_held_out=[5, 6])
    assert first.digest == json.loads(path.read_text())["manifest_sha256"]


def test_ik_does_not_clip_or_replace_requested_target():
    api = FrankaLiberoApi(_Env())
    calls = []
    api._solve_ik_with_prev = lambda position, quaternion_wxyz, prev_cfg=None: calls.append((position.copy(), quaternion_wxyz.copy())) or np.zeros(7)
    api.solve_ik(np.array([2.0, 1.0, 1.5]), np.array([1.0, 0.0, 0.0, 0.0]))
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], [2.0, 1.0, 1.4])
    np.testing.assert_allclose(calls[0][1], [1.0, 0.0, 0.0, 0.0])


def test_canonical_rpc_server_does_not_import_legacy_bridge():
    source = Path(__file__).parents[1].joinpath("roboforge/rpc_server.py").read_text()
    assert "LegacyAdapterBridge" not in source


def test_provider_passes_only_explicit_public_boundary_to_controller_runtime():
    class Deployment:
        instruction = "public task"
        sdk_index = {"methods": ["get_observation"]}

        def dispatch(self, method, arguments):
            return {"step": 0} if method == "observe" else {}

        def project_rpc_output(self, method, arguments, result):
            return result

        def canonical_embodied_state(self):
            return {"robot": {"joint_position": []}}

        def project_public_entities(self, tool_id, result):
            return []

        def sdk_consequence(self, method):
            return "READ_ONLY"

        def capability_consequence(self, tool_id):
            return "READ_ONLY"

    class Runtime:
        def execute(self, path, deployment, *, source_root=None):
            assert deployment.instruction == "public task"
            assert not hasattr(deployment, "env")
            assert not hasattr(deployment, "sim")
            assert not hasattr(deployment, "reset_case")
            return {"completed": True, "rpc_events": []}

    provider = LiberoProvider(Deployment(), Runtime())
    assert provider.task_info()["instruction"] == "public task"
    provider.runtime.execute(Path("controller.py"), provider, source_root=Path("."))
