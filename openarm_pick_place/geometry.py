from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import ObjectPose, Pose


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = Pose(translation, rotation)
    transform = np.eye(4)
    transform[:3, :3] = pose.rotation
    transform[:3, 3] = pose.position
    return transform


def inverse(transform: np.ndarray) -> np.ndarray:
    transform = validate_transform(transform)
    rotation = transform[:3, :3]
    result = np.eye(4)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ transform[:3, 3]
    return result


def transform_pose(transform: np.ndarray, pose: Pose) -> Pose:
    transform = validate_transform(transform)
    return Pose(transform[:3, :3] @ pose.position + transform[:3, 3], transform[:3, :3] @ pose.rotation)


def transform_object_pose(camera_pose: ObjectPose, base_from_camera: np.ndarray) -> ObjectPose:
    return ObjectPose(transform_pose(base_from_camera, camera_pose.pose), camera_pose.timestamp, camera_pose.valid_depth_points)


def validate_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=float)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError("invalid homogeneous transform bottom row")
    Pose(value[:3, 3], value[:3, :3])
    return value


def load_calibration(path: str | Path) -> tuple[np.ndarray, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"calibration_id", "camera_serial", "resolution", "base_from_camera"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"calibration missing: {', '.join(sorted(missing))}")
    return validate_transform(payload["base_from_camera"]), payload
