from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        position = _vector(self.position, 3, "position")
        rotation = np.asarray(self.rotation, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation must be orthonormal")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation", rotation)


@dataclass(frozen=True)
class ObjectPose:
    pose: Pose
    timestamp: float
    valid_depth_points: int = 0


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy, self.depth_scale)
        if not all(np.isfinite(values)) or self.fx <= 0 or self.fy <= 0 or self.depth_scale <= 0:
            raise ValueError("camera intrinsics and depth_scale must be finite and positive")


@dataclass(frozen=True)
class Workspace:
    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = _vector(self.minimum, 3, "workspace minimum")
        maximum = _vector(self.maximum, 3, "workspace maximum")
        if np.any(minimum >= maximum):
            raise ValueError("workspace minimum must be below maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def contains(self, position: np.ndarray) -> bool:
        point = _vector(position, 3, "position")
        return bool(np.all(point >= self.minimum) and np.all(point <= self.maximum))


@dataclass(frozen=True)
class GraspConfig:
    grasp_offset: np.ndarray
    approach_offset: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "grasp_offset", _vector(self.grasp_offset, 3, "grasp_offset"))
        object.__setattr__(self, "approach_offset", _vector(self.approach_offset, 3, "approach_offset"))


@dataclass
class TrialResult:
    success: bool = False
    final_state: str = "HOME"
    failure_reason: str = ""
    cycle_time: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
