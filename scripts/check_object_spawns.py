"""Spawn every registry object in every rest pose, let physics run, and measure.

For each case: build the scene, hold the arms at attention, step 1 s, and report how
far the object moved (mm) and turned (deg) from its spawn pose -- a correct spawn
rests exactly on the surface and does not move. Also renders the d435_head view
(RGB + the segmentation mask FoundationPose will get) and a close-up per case.

    python -m scripts.check_object_spawns                 # all cases
    python -m scripts.check_object_spawns --no-images     # numbers only

Writes artifacts/day1_spawns/summary.json and <case>_{d435,close,mask}.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from simulation.objects import Placement
from simulation.pick_place.scene import Scene

OUT_DIR = Path("artifacts") / "day1_spawns"
SETTLE_SECONDS = 1.0
MAX_DRIFT_MM = 2.0
MAX_TURN_DEG = 2.0
# The centre-basket layout (scripts/sweep_centre_basket.py): 10cm work platform,
# basket on the midline, the right arm's pick point on its side. The extra objects
# sit on the same side, >= 3cm clear of each other, the basket and the resting fist
# (a first try put the apple at (0.02, -0.22), under the right fist: it was shoved
# 14mm at t=0).
PLATFORM = 0.10
BASKET = (0.28, 0.0)
PICK = (0.28, -0.25)

SINGLE_CASES = [
    ("can", "upright", 0.0), ("can", "lying", 0.0), ("can", "lying", 45.0), ("can", "lying", 90.0),
    ("apple", "upright", 0.0), ("orange", "upright", 0.0),
    ("pear", "lying", 0.0), ("pear", "lying", 60.0),
    ("cracker_box", "upright", 0.0), ("cracker_box", "upright", 30.0),
    ("mustard_bottle", "upright", 0.0), ("mustard_bottle", "upright", 45.0), ("tuna_can", "upright", 0.0),
    ("tennis_ball", "upright", 0.0), ("gelatin_box", "upright", 0.0), ("gelatin_box", "upright", 60.0),
]
MULTI_CASE = {
    "pick": ("can", "lying", 30.0),
    "extras": [Placement("apple", (0.30, -0.42)), Placement("pear", (0.44, -0.22), "lying", 60.0)],
}


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def robot_touches(scene: Scene, key: str) -> bool:
    """The object is in contact with any arm or hand geom."""
    body = int(scene.model.joint(scene.object_joints[key]).bodyid[0])
    for contact in scene.data.contact[: scene.data.ncon]:
        for mine, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
            if int(scene.model.geom_bodyid[mine]) == body and scene._is_robot_geom(other):
                return True
    return False


def settle(scene: Scene, keys: list[str]) -> dict[str, dict[str, float]]:
    start = {key: scene.object_pose(key) for key in keys}
    touching_at_spawn = {key: robot_touches(scene, key) for key in keys}
    steps = int(round(SETTLE_SECONDS / scene.model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(scene.model, scene.data)
    result = {}
    for key in keys:
        end = scene.object_pose(key)
        result[key] = {
            "drift_mm": float(np.linalg.norm(end[:3, 3] - start[key][:3, 3]) * 1000.0),
            "turn_deg": rotation_angle_deg(start[key][:3, :3], end[:3, :3]),
            "z_mm": float(end[2, 3] * 1000.0),
            "robot_contact_at_spawn": touching_at_spawn[key],
        }
    return result


def render(scene: Scene, stem: str, focus: np.ndarray) -> dict[str, int]:
    """d435_head RGB, its segmentation (object pixels white) and a free close-up.
    Returns the object pixel count per key, as FoundationPose's mask would see it."""
    import imageio.v3 as iio  # only needed for images

    renderer = mujoco.Renderer(scene.model, height=480, width=640)
    renderer.update_scene(scene.data, camera="d435_head")
    iio.imwrite(OUT_DIR / f"{stem}_d435.png", renderer.render())
    renderer.enable_segmentation_rendering()
    renderer.update_scene(scene.data, camera="d435_head")
    segmentation = renderer.render()
    renderer.disable_segmentation_rendering()
    geom_ids = segmentation[..., 0]
    pixels, mask = {}, np.zeros(geom_ids.shape, dtype=np.uint8)
    for key, joint_name in scene.object_joints.items():
        body = int(scene.model.joint(joint_name).bodyid[0])
        # The visual mesh (group 2) is what the camera sees; the collision geom is
        # transparent. Either belonging to the body counts.
        ids = [g for g in range(scene.model.ngeom) if int(scene.model.geom_bodyid[g]) == body]
        hit = np.isin(geom_ids, ids)
        pixels[key] = int(hit.sum())
        mask[hit] = 255
    iio.imwrite(OUT_DIR / f"{stem}_mask.png", mask)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = focus
    camera.distance, camera.azimuth, camera.elevation = 0.45, 135.0, -30.0
    renderer.update_scene(scene.data, camera=camera)
    iio.imwrite(OUT_DIR / f"{stem}_close.png", renderer.render())
    renderer.close()
    return pixels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        (f"{key}_{pose}_{int(yaw)}", dict(pick_object=key, pick_pose=pose, pick_yaw_deg=yaw))
        for key, pose, yaw in SINGLE_CASES
    ]
    key, pose, yaw = MULTI_CASE["pick"]
    cases.append((
        "multi_can_apple_pear",
        dict(pick_object=key, pick_pose=pose, pick_yaw_deg=yaw, extra_objects=MULTI_CASE["extras"]),
    ))

    summary, failures = {}, 0
    print(f"{'case':28s} {'object':7s} {'drift mm':>9s} {'turn deg':>9s} {'z mm':>7s} {'mask px':>8s}")
    for stem, kwargs in cases:
        scene = Scene(pick_position=PICK, basket_position=BASKET, work_platform_height=PLATFORM, **kwargs)
        keys = list(scene.object_joints)
        measured = settle(scene, keys)
        pixels = {} if args.no_images else render(scene, stem, scene.object_pose()[:3, 3])
        for key, row in measured.items():
            row["mask_px"] = pixels.get(key)
            ok = (row["drift_mm"] <= MAX_DRIFT_MM and row["turn_deg"] <= MAX_TURN_DEG
                  and not row["robot_contact_at_spawn"])
            row["ok"] = ok
            failures += not ok
            print(f"{stem:28s} {key:7s} {row['drift_mm']:9.2f} {row['turn_deg']:9.2f} {row['z_mm']:7.1f} "
                  f"{row['mask_px'] if row['mask_px'] is not None else '-':>8}  {'OK' if ok else 'MOVED'}")
        summary[stem] = measured
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{failures} object(s) touched the robot at spawn or moved more than {MAX_DRIFT_MM}mm / {MAX_TURN_DEG}deg in {SETTLE_SECONDS}s")


if __name__ == "__main__":
    main()
