from __future__ import annotations

import time

import numpy as np

from .models import CameraIntrinsics, ObjectPose, Pose


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=float) / 255.0
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    maximum, minimum = image.max(axis=2), image.min(axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 0
    red = nonzero & (maximum == image[:, :, 0])
    green = nonzero & (maximum == image[:, :, 1])
    blue = nonzero & (maximum == image[:, :, 2])
    red_hue = np.divide(image[:, :, 1] - image[:, :, 2], delta, out=np.zeros_like(delta), where=nonzero)
    green_hue = np.divide(image[:, :, 2] - image[:, :, 0], delta, out=np.zeros_like(delta), where=nonzero)
    blue_hue = np.divide(image[:, :, 0] - image[:, :, 1], delta, out=np.zeros_like(delta), where=nonzero)
    hue[red] = red_hue[red] % 6
    hue[green] = green_hue[green] + 2
    hue[blue] = blue_hue[blue] + 4
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)
    return np.stack((hue * 30, saturation * 255, maximum * 255), axis=2)


def segment_hsv(rgb: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    hsv = rgb_to_hsv(rgb)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower > upper):
        raise ValueError("HSV bounds must be ordered 3-vectors")
    return np.all((hsv >= lower) & (hsv <= upper), axis=2)


def estimate_object_pose(
    rgb: np.ndarray,
    aligned_depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    hsv_lower: np.ndarray,
    hsv_upper: np.ndarray,
    *,
    timestamp: float | None = None,
    minimum_points: int = 20,
) -> ObjectPose | None:
    mask = segment_hsv(rgb, hsv_lower, hsv_upper)
    depth = np.asarray(aligned_depth, dtype=float) * intrinsics.depth_scale
    if depth.shape != mask.shape:
        raise ValueError("aligned depth dimensions must match RGB")
    valid = mask & np.isfinite(depth) & (depth > 0)
    rows, columns = np.nonzero(valid)
    if rows.size < minimum_points:
        return None
    values = depth[valid]
    median = np.median(values)
    deviation = np.median(np.abs(values - median))
    keep = np.ones(values.shape, dtype=bool) if deviation == 0 else np.abs(values - median) <= 3 * deviation
    rows, columns, values = rows[keep], columns[keep], values[keep]
    if values.size < minimum_points:
        return None
    z = float(np.median(values))
    u, v = float(np.median(columns)), float(np.median(rows))
    position = np.array(((u - intrinsics.cx) * z / intrinsics.fx, (v - intrinsics.cy) * z / intrinsics.fy, z))
    centered = np.column_stack((columns - u, rows - v))
    covariance = centered.T @ centered
    axis = np.linalg.eigh(covariance)[1][:, -1]
    yaw = float(np.arctan2(axis[1], axis[0]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))
    return ObjectPose(Pose(position, rotation), time.time() if timestamp is None else timestamp, int(values.size))
