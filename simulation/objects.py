"""Registry of the YCB objects the scene can hold, and how each one rests on a table.

One entry per object: which asset folder, what it collides as, mass and friction,
its symmetry (the grasp library and the pose-error metric both need it) and the
named rest poses it can be spawned in. A rest pose is an orientation *before* the
spawn yaw; the spawn height is derived from the collision geometry in that
orientation, so every object starts exactly touching the surface (never dropped,
never sunk in).

Body frame = the mesh's own OBJ frame for every object. That is also the frame
FoundationPose reports its pose in when it is given the same .obj, so the ground
truth `T_world_body` and the estimate are directly comparable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(os.environ.get("OPENARM_PROJECT_ROOT", Path(__file__).parents[1]))
if not (_PROJECT_ROOT / "assets/ycb").is_dir():
    _PROJECT_ROOT = Path.cwd()
YCB_ASSET_ROOT = _PROJECT_ROOT / "assets" / "ycb"

_SQRT_HALF = float(np.sqrt(0.5))
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
# +90 degrees about x: the object's +y axis turns to world +z, +z to world -y.
Y_UP_QUAT = (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0)
# +90 degrees about x applied to a z-axis object: its axis ends up along world -y,
# i.e. lying on its side.
LYING_QUAT = Y_UP_QUAT


@dataclass(frozen=True)
class ObjectSpec:
    key: str
    asset: str                       # folder under assets/ycb (without the ycb_ prefix)
    collision: str                   # "cylinder" (analytic) or "mesh" (convex hull of the OBJ)
    mass: float                      # kg
    symmetry: str                    # "axial" (about body z), "spherical" or "none"
    poses: dict[str, tuple[float, float, float, float]]
    cylinder: tuple[float, float] = (0.0, 0.0)   # (radius, half-height) for "cylinder"
    # Cylinder centre in the body (OBJ) frame: YCB scans are not centred on their origin.
    collision_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    note: str = ""
    # Visual mesh offset in the body frame. Only the can needs one: its collision
    # cylinder is a little taller than the mesh (see five_finger_model).
    visual_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def mesh_file(self) -> Path:
        return YCB_ASSET_ROOT / f"ycb_{self.asset}" / "meshes" / f"{self.asset}.obj"

    @property
    def texture_file(self) -> Path:
        return YCB_ASSET_ROOT / f"ycb_{self.asset}" / "textures" / f"{self.asset}.png"


# Masses: the YCB object set's published values, except the can, which keeps the
# 0.2 kg the existing pick-place trials are calibrated against (the real, full can
# is 0.349 kg -- revisit before sim-to-real).
OBJECTS: dict[str, ObjectSpec] = {
    "can": ObjectSpec(
        key="can", asset="tomato_soup_can", collision="cylinder", mass=0.2, symmetry="axial",
        poses={"upright": IDENTITY_QUAT, "lying": LYING_QUAT},
        cylinder=(0.0354, 0.05), friction=(1.2, 0.02, 0.002),
        visual_offset=(0.0, 0.0, 0.05 - 0.0516),
        note="collision cylinder r 35.4mm, half-height 50mm (mesh 34.0 / 50.9mm)",
    ),
    "apple": ObjectSpec(
        key="apple", asset="apple", collision="mesh", mass=0.068, symmetry="spherical",
        poses={"upright": IDENTITY_QUAT}, note="75 x 75 x 72 mm",
    ),
    "orange": ObjectSpec(
        key="orange", asset="orange", collision="mesh", mass=0.047, symmetry="spherical",
        poses={"upright": IDENTITY_QUAT}, note="72 x 74 x 71 mm",
    ),
    # The user's object set (2026-09-28): YCB 003, 005 (the can above), 006, 007, 010.
    # Meshes are the YCB google_16k scans from datasets_project/mujoco-ycb-dataset,
    # copied unmodified (their own origin, z up as scanned; the spawn height is derived
    # from the vertices, so an off-centre origin is fine). Masses from the YCB set.
    # Cracker box: 72 (x) x 164 (y) x 213 (z) mm. Only standing: lying flat both
    # horizontal sides exceed the open jaw (~113 mm), lying on its side 213 mm does not
    # fit the 180 mm basket.
    "cracker_box": ObjectSpec(
        key="cracker_box", asset="cracker_box", collision="mesh", mass=0.411, symmetry="none",
        poses={"upright": IDENTITY_QUAT}, note="YCB 003, 72 x 164 x 213 mm",
    ),
    # Mustard bottle: 97 (x) x 67 (y) x 191 (z) mm. Standing only for the 180 mm basket.
    "mustard_bottle": ObjectSpec(
        key="mustard_bottle", asset="mustard_bottle", collision="mesh", mass=0.603, symmetry="none",
        poses={"upright": IDENTITY_QUAT}, note="YCB 006, 97 x 67 x 191 mm",
    ),
    # Tuna can: 86 mm across, 34 mm tall; it rests flat. Collides as a cylinder over
    # the scan's AABB: its convex hull on the work platform (a box) sank 6 mm on one
    # edge and crept 1.8 mm/s (MuJoCo mesh-box contact, normal 2.5 deg off vertical);
    # the cylinder rests like the soup can does.
    "tuna_can": ObjectSpec(
        key="tuna_can", asset="tuna_fish_can", collision="cylinder", mass=0.171, symmetry="axial",
        poses={"upright": IDENTITY_QUAT}, cylinder=(0.0428, 0.01675),
        collision_offset=(-0.0260, -0.0221, 0.0136), note="YCB 007, 86 x 86 x 34 mm",
    ),
    # Peach (YCB 015, from the P-161 set like the apple/orange): 62 x 63 x 59 mm, 33 g.
    "peach": ObjectSpec(
        key="peach", asset="peach", collision="mesh", mass=0.033, symmetry="spherical",
        poses={"upright": IDENTITY_QUAT}, note="YCB 015, 62 x 63 x 59 mm",
    ),
    # Tennis ball (YCB 056): 67 mm sphere, 58 g.
    "tennis_ball": ObjectSpec(
        key="tennis_ball", asset="tennis_ball", collision="mesh", mass=0.058, symmetry="spherical",
        poses={"upright": IDENTITY_QUAT}, note="YCB 056, 67 mm sphere",
    ),
    # Gelatin box (YCB 009), lying flat as scanned: 72.9 x 89.3 mm top face, turned
    # 13.25 deg about z in the scan's frame (minimum-area rectangle of the vertices),
    # 30 mm tall, 97 g.
    "gelatin_box": ObjectSpec(
        key="gelatin_box", asset="gelatin_box", collision="mesh", mass=0.097, symmetry="none",
        poses={"upright": IDENTITY_QUAT}, note="YCB 009, 73 x 89 x 30 mm, lying flat",
    ),
    # The pear's OBJ has its long axis along +y, wide end at -y: as shipped it lies
    # on its side, which is also the only way it rests. Stood on its wide end it
    # tips over (measured: 90 degrees within 1s, scripts/check_object_spawns.py).
    "pear": ObjectSpec(
        key="pear", asset="pear", collision="mesh", mass=0.049, symmetry="none",
        poses={"lying": IDENTITY_QUAT}, note="67 x 100 x 66 mm, wide end at -y",
    ),
}


@dataclass(frozen=True)
class Placement:
    """One object in the scene: which object, where on the surface (`xy` = the centre
    of its footprint, not its body origin), in which rest pose, turned by `yaw_deg`
    about the world vertical."""

    key: str
    xy: tuple[float, float]
    pose: str = "upright"
    yaw_deg: float = 0.0
    # Instance name, for more than one of the same object in a scene ("can_2");
    # defaults to the registry key.
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or self.key

    def __post_init__(self) -> None:
        spec = OBJECTS.get(self.key)
        if spec is None:
            raise ValueError(f"unknown object {self.key!r}; known: {sorted(OBJECTS)}")
        if self.pose not in spec.poses:
            raise ValueError(f"{self.key} has no rest pose {self.pose!r}; known: {sorted(spec.poses)}")

    @property
    def spec(self) -> ObjectSpec:
        return OBJECTS[self.key]


# ---------------------------------------------------------------------- geometry
def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_matrix(rotation) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix (MuJoCo's own conversion)."""
    import mujoco

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).reshape(9))
    return quat


def yaw_quat(yaw_deg: float) -> np.ndarray:
    half = 0.5 * np.deg2rad(yaw_deg)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


@lru_cache(maxsize=None)
def mesh_vertices(key: str) -> np.ndarray:
    """OBJ vertex positions (body frame, metres)."""
    path = OBJECTS[key].mesh_file
    if not path.is_file():
        raise FileNotFoundError(f"YCB mesh asset missing: {path}")
    with path.open() as handle:
        rows = [line.split()[1:4] for line in handle if line.startswith("v ")]
    return np.asarray(rows, dtype=float)


def collision_points(key: str) -> np.ndarray:
    """Points whose convex hull is the collision shape (body frame). For the
    cylinder: its two rim circles, sampled finely enough for a height query."""
    spec = OBJECTS[key]
    if spec.collision == "mesh":
        return mesh_vertices(key)
    radius, half_height = spec.cylinder
    angle = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    ring = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    rims = np.vstack([np.column_stack([ring, np.full(len(ring), z)]) for z in (-half_height, half_height)])
    return rims + np.asarray(spec.collision_offset)


def spawn_quat(placement: Placement) -> np.ndarray:
    """World orientation (w, x, y, z): the rest pose, then the yaw about world z."""
    return quat_mul(yaw_quat(placement.yaw_deg), placement.spec.poses[placement.pose])


def spawn_height(placement: Placement) -> float:
    """Height of the body origin above the supporting surface so the lowest point of
    the collision shape just touches it."""
    rotation = quat_to_matrix(spawn_quat(placement))
    return float(-(collision_points(placement.key) @ rotation.T)[:, 2].min())


def _footprint(placement: Placement) -> tuple[np.ndarray, float]:
    """(centre offset of the footprint from the body origin, world xy; radius of the
    footprint's bounding circle about that centre) in the spawn orientation."""
    points = (collision_points(placement.key) @ quat_to_matrix(spawn_quat(placement)).T)[:, :2]
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    return centre, float(np.linalg.norm(points - centre, axis=1).max())


def spawn_origin_xy(placement: Placement) -> np.ndarray:
    """World xy of the body origin that puts the footprint's centre at `placement.xy`
    (YCB scans are not centred on their origin: the tuna can's is 34 mm off)."""
    return np.asarray(placement.xy, dtype=float) - _footprint(placement)[0]


def footprint_radius(placement: Placement | str) -> float:
    """Radius of the footprint's bounding circle about its centre (spacing checks).
    A bare key means the object as spawned upright/first pose, unturned."""
    if isinstance(placement, str):
        placement = Placement(placement, (0.0, 0.0), next(iter(OBJECTS[placement].poses)))
    return _footprint(placement)[1]


def check_spacing(placements, min_gap: float = 0.005) -> None:
    """Every pair of objects keeps at least `min_gap` between their bounding
    circles (conservative: a lying can's circle is 61 mm for a 35 x 50 mm half
    footprint); the planner's collision checks decide whether the hand fits."""
    for i, a in enumerate(placements):
        for b in placements[i + 1 :]:
            distance = float(np.hypot(a.xy[0] - b.xy[0], a.xy[1] - b.xy[1]))
            needed = footprint_radius(a) + footprint_radius(b) + min_gap
            if distance < needed:
                raise ValueError(
                    f"{a.key} and {b.key} are {distance*100:.1f}cm apart; need >= {needed*100:.1f}cm"
                )


def geometric_center(key: str) -> np.ndarray:
    """Centre of the collision shape's bounding box in the body (OBJ) frame: ~0 for
    the centred meshes (can, fruit), 34 mm off for the tuna can's scan."""
    points = collision_points(key)
    return 0.5 * (points.min(axis=0) + points.max(axis=0))


def body_name(key: str) -> str:
    return f"obj_{key}"


__all__ = [
    "OBJECTS", "ObjectSpec", "Placement", "body_name", "check_spacing", "collision_points",
    "footprint_radius", "mesh_vertices", "quat_mul", "quat_to_matrix", "spawn_height", "spawn_quat",
    "yaw_quat",
]
