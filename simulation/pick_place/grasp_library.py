"""Grasp library: from an object's 6D pose to what the planner needs to take it.

The library (config/grasp_library.yaml) holds, per object and rest pose, geometry in
the object's own frame: where the jaw closes, which object axis the hand must wrap
around, the object's width/height/footprint and what "not tilted" means for it.
`select_grasp(key, T_world_object)` picks the entry whose rest-pose condition the pose
satisfies and turns it into a `GraspTarget` in world terms:

    centre_world      -> the point the planner centres the jaw on
    jaw_heading_deg   -> required world heading of the thumb->fingers line (mod 180),
                         or None when any heading will do (round seen from above)
    width / height    -> jaw clearance check, carry height, set-down

The planner turns the heading into hand yaws (GraspPlanner._yaw_candidates); the
wrist target is still derived from the jaw, as for the original can grasp.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

LIBRARY_PATH = Path(__file__).resolve().parents[2] / "config" / "grasp_library.yaml"
# A rest-pose axis counts as vertical / horizontal within this of the ideal.
REST_POSE_TOLERANCE_DEG = 45.0
UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class AxisRule:
    axis: tuple[float, float, float]   # object frame
    is_: str                           # "vertical" | "horizontal"

    def deviation_deg(self, rotation: np.ndarray) -> float:
        """How far the object axis is from the rule (0 = exactly vertical/horizontal)."""
        axis = rotation @ np.asarray(self.axis, dtype=float)
        axis /= np.linalg.norm(axis)
        up = float(np.clip(axis @ UP, -1.0, 1.0))
        if self.is_ == "vertical":
            return float(np.degrees(np.arccos(up)))  # upside down is 180, not 0
        return float(np.degrees(np.arcsin(abs(up))))


@dataclass(frozen=True)
class GraspEntry:
    object_key: str
    name: str
    center: tuple[float, float, float]
    width: float
    height: float
    footprint: float
    when: AxisRule | None = None
    jaw_across: tuple[float, float, float] | None = None
    keep: AxisRule | None = None
    # Jaw centre height above `center` (m); None = the planner's own (GRASP_HEIGHT_BIAS,
    # the upright-can top grasp).
    height_bias: float | None = None
    # Jaw offsets off square to `jaw_across` the planner may try (deg); None = the
    # planner's C.GRASP_JAW_OFFSETS_DEG (fine around a cylinder). Flat-sided objects
    # take [0]: 15-30 deg off, the fingers landed on the mustard bottle's edges and it
    # stayed on the table while the hand rose 44 mm (2026-09-28).
    jaw_offsets_deg: tuple[float, ...] | None = None
    # Finger closing order: "fingers_first" (four fingers to light contact, then the
    # thumb -- the can pipeline) or "together" (thumb with the fingers: a ball rolled
    # 10 cm away from fingers closing first, 2026-09-28).
    close: str = "fingers_first"


@dataclass
class GraspTarget:
    """One grasp resolved against a pose. `pose` is the T_world_object it came from."""

    entry: GraspEntry
    pose: np.ndarray
    center_world: np.ndarray
    jaw_heading_deg: float | None

    @property
    def object_key(self) -> str:
        return self.entry.object_key

    @property
    def name(self) -> str:
        return self.entry.name

    def tilt_deg(self, rotation: np.ndarray) -> float:
        """Tilt of the object *for this grasp's rest pose*: 0 while it rests as it
        should (can upright: axis vertical; can/pear lying: axis horizontal; fruit:
        always 0 -- a sphere has no tilt)."""
        return 0.0 if self.entry.keep is None else self.entry.keep.deviation_deg(rotation)


def _axis_rule(raw: dict | None) -> AxisRule | None:
    if raw is None:
        return None
    if raw.get("is") not in ("vertical", "horizontal"):
        raise ValueError(f"axis rule needs is: vertical|horizontal, got {raw}")
    return AxisRule(tuple(float(v) for v in raw["axis"]), raw["is"])


@lru_cache(maxsize=None)
def load_library(path: Path = LIBRARY_PATH) -> dict[str, tuple[GraspEntry, ...]]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    library = {}
    for key, entries in raw.items():
        parsed = []
        for item in entries:
            jaw = item.get("jaw_across")
            parsed.append(GraspEntry(
                object_key=key, name=item["name"], center=tuple(float(v) for v in item["center"]),
                width=float(item["width"]), height=float(item["height"]), footprint=float(item["footprint"]),
                when=_axis_rule(item.get("when")), jaw_across=None if jaw is None else tuple(float(v) for v in jaw),
                keep=_axis_rule(item.get("keep")),
                height_bias=None if item.get("height_bias") is None else float(item["height_bias"]),
                jaw_offsets_deg=None if item.get("jaw_offsets_deg") is None
                else tuple(float(v) for v in item["jaw_offsets_deg"]),
                close=_close_order(item.get("close", "fingers_first")),
            ))
        library[key] = tuple(parsed)
    return library


def _close_order(value: str) -> str:
    if value not in ("fingers_first", "together"):
        raise ValueError(f"close must be fingers_first or together, got {value!r}")
    return value


def jaw_heading_deg(rotation: np.ndarray, jaw_across) -> float | None:
    """World heading (deg, in [0, 180)) of the horizontal line perpendicular to the
    object axis `jaw_across`; None when that axis is (near) vertical, i.e. every
    horizontal heading is perpendicular to it."""
    if jaw_across is None:
        return None
    axis = rotation @ np.asarray(jaw_across, dtype=float)
    across = np.cross(UP, axis)  # horizontal and perpendicular to the axis
    if np.linalg.norm(across) < np.sin(np.radians(10.0)) * np.linalg.norm(axis):
        return None
    return float(np.degrees(np.arctan2(across[1], across[0])) % 180.0)


def select_grasp(object_key: str, pose: np.ndarray, library: dict | None = None) -> GraspTarget:
    """The first library grasp for `object_key` whose rest-pose condition `pose`
    (4x4 T_world_object) satisfies. Raises when the object is unknown or resting in a
    way no entry covers (e.g. an upright pear) -- never guesses."""
    return select_grasps(object_key, pose, library)[0]


def select_grasps(object_key: str, pose: np.ndarray, library: dict | None = None) -> list[GraspTarget]:
    """Every library grasp whose rest-pose condition `pose` satisfies, in library
    order (the preferred one first; the planner falls back down the list when a
    grasp has no reachable heading). Raises when none does."""
    library = load_library() if library is None else library
    entries = library.get(object_key)
    if not entries:
        raise RuntimeError(f"no grasp library entry for object {object_key!r}")
    pose = np.asarray(pose, dtype=float)
    rotation = pose[:3, :3]
    reasons, targets = [], []
    for entry in entries:
        if entry.when is not None:
            deviation = entry.when.deviation_deg(rotation)
            if deviation > REST_POSE_TOLERANCE_DEG:
                reasons.append(f"{entry.name}: axis {deviation:.0f}deg from {entry.when.is_}")
                continue
        center = pose[:3, :3] @ np.asarray(entry.center) + pose[:3, 3]
        targets.append(GraspTarget(entry, pose.copy(), center, jaw_heading_deg(rotation, entry.jaw_across)))
    if not targets:
        raise RuntimeError(f"no grasp for {object_key} in this pose: " + "; ".join(reasons))
    return targets
