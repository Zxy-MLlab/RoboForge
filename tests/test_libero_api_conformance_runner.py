from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from scripts.validate_libero_api_conformance import (
    _manifest_digest,
    _npy64,
    _parse_molmo_point,
    _validate_graspnet_response,
    _validate_joint_response,
    _validate_sam3_response,
    _validated_resume,
)


def test_service_payload_validators_require_semantic_arrays() -> None:
    grasps = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    validated = _validate_graspnet_response(
        {
            "grasps_base64": _npy64(grasps),
            "scores_base64": _npy64(np.array([0.8, 0.7], dtype=np.float32)),
            "contact_pts_base64": _npy64(np.ones((2, 3), dtype=np.float32)),
        }
    )
    assert validated["num_grasps"] == 2
    assert _validate_joint_response({"joint_positions": [0.0] * 7})["finite"]

    with pytest.raises(ValueError, match="empty"):
        _validate_graspnet_response(
            {
                "grasps_base64": _npy64(np.empty((0, 4, 4))),
                "scores_base64": _npy64(np.empty((0,))),
                "contact_pts_base64": _npy64(np.empty((0, 3))),
            }
        )
    with pytest.raises(ValueError, match="invalid joint solution"):
        _validate_joint_response({"joint_positions": [float("nan")] * 7})


def test_sam3_validator_requires_nonempty_well_formed_mask() -> None:
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    body = {
        "results": [
            {
                "mask_base64": base64.b64encode(mask.tobytes()).decode(),
                "shape": [2, 2],
                "box": [0.0, 0.0, 2.0, 2.0],
                "score": 0.75,
            }
        ]
    }
    assert _validate_sam3_response(body)["num_valid_nonempty_masks"] == 1
    body["results"][0]["mask_base64"] = base64.b64encode(np.zeros_like(mask).tobytes()).decode()
    with pytest.raises(ValueError, match="no non-empty"):
        _validate_sam3_response(body)


def test_molmo_parser_maps_upstream_formats_to_pixels() -> None:
    assert _parse_molmo_point(
        '<points coords="1 1 750 500">object</points>', width=200, height=100
    ) == (150, 50)
    assert _parse_molmo_point(
        '<point x="25" y="75">object</point>', width=200, height=100
    ) == (50, 75)
    assert _parse_molmo_point("not a point", width=200, height=100) is None


def test_resume_requires_digest_and_identical_observation(tmp_path) -> None:
    manifest = {
        "task": 0,
        "state": 0,
        "controller_mode": "JOINT_POSITION",
        "observation_fingerprint": "abc",
        "rows": [],
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    path = tmp_path / "phase.json"
    path.write_text(json.dumps(manifest))
    assert _validated_resume(
        path, task=0, state=0, observation_fingerprint="abc"
    )["manifest_sha256"] == manifest["manifest_sha256"]

    tampered = dict(manifest)
    tampered["state"] = 1
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="digest mismatch"):
        _validated_resume(path, task=0, state=0, observation_fingerprint="abc")

