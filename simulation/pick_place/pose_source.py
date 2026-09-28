"""The one seam between perception and manipulation: `get_object_pose`.

    T_world_object = get_object_pose(scene, key, backend)

Backends:
    gt              simulator state -- baseline and debugging only, never a detector
    color           the RGB-D colour detector (vision_detector.py): position only, so
                    the orientation is assumed upright (the can it was written for)
    foundationpose  6D model-based pose from FoundationPose in WSL (simulation/fp_bridge.py)

Everything downstream (grasp library, planner) sees only the 4x4 pose, so swapping the
backend changes nothing else.
"""

from __future__ import annotations

import numpy as np

BACKENDS = ("gt", "color", "foundationpose")


def get_object_pose(scene, key: str | None = None, backend: str = "gt", camera: str = "d435_head",
                    mask: np.ndarray | None = None) -> np.ndarray:
    """`mask`: the object's pixels from the detector (object_detector.py); used by
    FoundationPose, which otherwise falls back to the simulator's segmentation."""
    key = key or scene.pick_object
    if backend == "gt":
        return scene.object_pose(key)
    if backend == "color":
        from simulation.vision_detector import VisionDetector

        if scene.object_types.get(key, key) != "can":
            raise RuntimeError(f"the colour detector only finds the red can, not {key!r}")
        result = VisionDetector(scene.model, camera).detect_object(scene.data, render_annotation=False)
        if not result.found:
            raise RuntimeError(f"perception failed: {key} not found in camera '{camera}' ({result.pixel_count} px)")
        pose = np.eye(4)
        pose[:3, 3] = result.pos_world
        return pose
    if backend == "foundationpose":
        from simulation.fp_bridge import estimate_pose

        return estimate_pose(scene, key, camera, mask=mask)
    raise ValueError(f"unknown pose backend {backend!r}; known: {BACKENDS}")


def pose_error(estimate: np.ndarray, truth: np.ndarray, symmetry: str = "none", axis=(0.0, 0.0, 1.0)) -> tuple[float, float]:
    """(translation error m, rotation error deg), symmetry-aware: an axially symmetric
    object is scored on its symmetry axis only, a spherical one on position only."""
    translation = float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]))
    if symmetry == "spherical":
        return translation, 0.0
    if symmetry == "axial":
        a = estimate[:3, :3] @ np.asarray(axis)
        b = truth[:3, :3] @ np.asarray(axis)
        return translation, float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), -1.0, 1.0))))
    relative = estimate[:3, :3].T @ truth[:3, :3]
    return translation, float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
