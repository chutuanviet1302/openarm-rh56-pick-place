from __future__ import annotations

import numpy as np

from .models import GraspConfig, Pose, Workspace


def generate_grasp(object_pose: Pose, config: GraspConfig) -> tuple[Pose, Pose]:
    grasp = Pose(object_pose.position + object_pose.rotation @ config.grasp_offset, object_pose.rotation)
    pregrasp = Pose(grasp.position + object_pose.rotation @ config.approach_offset, grasp.rotation)
    return pregrasp, grasp


def pick_place_waypoints(object_pose: Pose, place_pose: Pose, config: GraspConfig, lift_height: float = 0.08) -> dict[str, Pose]:
    if lift_height <= 0:
        raise ValueError("lift_height must be positive")
    pregrasp, grasp = generate_grasp(object_pose, config)
    lift = Pose(grasp.position + [0, 0, lift_height], grasp.rotation)
    preplace = Pose(place_pose.position + [0, 0, lift_height], place_pose.rotation)
    return {
        "pregrasp": pregrasp,
        "grasp": grasp,
        "lift": lift,
        "preplace": preplace,
        "place": place_pose,
        "retreat": preplace,
    }


def joint_trajectory(start: np.ndarray, target: np.ndarray, max_velocity: np.ndarray, dt: float) -> np.ndarray:
    start, target, velocity = map(lambda value: np.asarray(value, dtype=float), (start, target, max_velocity))
    if start.ndim != 1 or target.shape != start.shape or velocity.shape != start.shape:
        raise ValueError("joint vectors must have the same one-dimensional shape")
    if dt <= 0 or np.any(velocity <= 0) or not np.all(np.isfinite([*start, *target, *velocity, dt])):
        raise ValueError("trajectory inputs must be finite and velocity/dt positive")
    duration = float(np.max(np.abs(target - start) / velocity))
    steps = max(1, int(np.ceil(duration / dt)))
    return np.linspace(start, target, steps + 1)


def validate_target(pose: Pose, workspace: Workspace, table_height: float, clearance: float) -> None:
    if not workspace.contains(pose.position):
        raise ValueError("target outside safe workspace")
    if pose.position[2] < table_height + clearance:
        raise ValueError("target violates table clearance")
