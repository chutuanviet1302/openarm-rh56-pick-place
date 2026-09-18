"""Find the reference postures the planner is built on, by sampling forward kinematics.

Two postures are derived here rather than typed in:

  grasp     -- right arm top-grasp, 20/25/30 degrees off vertical, with the jaw centre
               on the upper half of a can standing on the floor.
  attention -- both arms symmetric, fists held forward over the table with a straight
               wrist, well clear of the table and of the pick region
               (config.ATTENTION_RIGHT; the left arm mirrors it).

Run after anything that changes the arm or the hand mount, paste the chosen rows
into config.py, then run the tests.

    python -m scripts.sweep_postures            # both sweeps
    python -m scripts.sweep_postures grasp      # one of them
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

from simulation.five_finger_model import TABLE_TOP_Z, OBJECT_HALF_HEIGHT
from simulation.pick_place.config import (
    ARM_JOINTS,
    GRASP_HEIGHT_BIAS,
    MIN_FLOOR_CLEARANCE,
    MIN_JOINT_MARGIN_DEG,
    TOP_GRASP_TILT_CANDIDATES_DEG,
)
from simulation.pick_place.planner import GraspPlanner
from simulation.pick_place.scene import Scene

CAN_WAIST_Z = TABLE_TOP_Z + OBJECT_HALF_HEIGHT
np.set_printoptions(precision=3, suppress=True)


def _limits(model, side):
    lo = np.array([model.joint(n).range[0] for n in ARM_JOINTS[side]])
    hi = np.array([model.joint(n).range[1] for n in ARM_JOINTS[side]])
    return lo, hi


def sweep_grasp(samples: int = 500_000, seed: int = 1, show: int = 12):
    scene = Scene()
    model, data = scene.model, mujoco.MjData(scene.model)
    data.qpos[:] = scene.data.qpos
    planner = GraspPlanner(scene)
    local_fingers, local_thumb = planner.local_jaw_offsets()
    q = scene.arm_qpos["right"]
    site = scene.ee_site_id["right"]
    tips = [model.site(f"inspire_right_right_{f}_tip").id for f in ("index", "middle", "ring", "pinky", "thumb")]
    lo, hi = _limits(model, "right")
    rng = np.random.default_rng(seed)
    hits = []
    reachable_tilts = []
    for _ in range(samples):
        v = rng.uniform(lo, hi)
        data.qpos[q] = v
        mujoco.mj_kinematics(model, data)
        wrist = data.site_xpos[site]
        rotation = data.site_xmat[site].reshape(3, 3)
        fingers, thumb = rotation @ local_fingers, rotation @ local_thumb
        jaw = wrist + 0.5 * (fingers + thumb)
        obj = jaw - np.array([0.0, 0.0, GRASP_HEIGHT_BIAS])
        if abs(obj[2] - CAN_WAIST_Z) > 0.012:
            continue
        if not (0.18 < obj[0] < 0.55 and -0.50 < obj[1] < -0.12):
            continue
        finger_dir = fingers / np.linalg.norm(fingers)
        tilt = np.degrees(np.arccos(np.clip(-finger_dir[2], -1.0, 1.0)))
        reachable_tilts.append(tilt)
        candidate = min(TOP_GRASP_TILT_CANDIDATES_DEG, key=lambda angle: abs(angle - tilt))
        if abs(candidate - tilt) > 2.5:
            continue
        jaw_line = thumb - fingers
        if abs(jaw_line[2]) > 0.025:
            continue
        lowest = float(data.site_xpos[tips].min(0)[2])
        if lowest < TABLE_TOP_Z + MIN_FLOOR_CLEARANCE:
            continue
        margin = np.min(np.minimum(v - lo, hi - v))
        if margin < np.deg2rad(MIN_JOINT_MARGIN_DEG):
            continue
        score = abs(candidate - tilt) - 2.0 * margin + 0.5 * (abs(v[5]) + abs(v[6]))
        hits.append((score, v.copy(), obj.copy(), finger_dir.copy(), tilt, lowest, np.degrees(margin)))
    hits.sort(key=lambda h: h[0])
    if reachable_tilts:
        print(f"reachable jaw samples: {len(reachable_tilts)}, tilt range {min(reachable_tilts):.1f}..{max(reachable_tilts):.1f} deg")
    print(f"grasp: {len(hits)} candidates (lower score = target tilt + larger joint margin)")
    print(f"{'score':>6} {'joints (7)':<50} {'object A (x y z)':<24} {'finger dir':<24} tilt clear margin")
    for score, v, obj, fd, tilt, lowest, margin in hits[:show]:
        print(f"{score:6.2f} {np.array2string(v, precision=3):<50} {np.array2string(obj):<24} {np.array2string(fd):<24} {tilt:4.1f} {lowest:5.3f} {margin:4.1f}")
    return hits


def sweep_attention(samples: int = 400_000, seed: int = 4, show: int = 8):
    scene = Scene()
    model, data = scene.model, mujoco.MjData(scene.model)
    data.qpos[:] = scene.data.qpos  # fists closed
    q = scene.arm_qpos["right"]
    palm_body = model.body("inspire_right_base").id
    hand_geoms = [g for g in range(model.ngeom) if "inspire_right" in (model.body(model.geom_bodyid[g]).name or "")]
    tips = [model.site(f"inspire_right_right_{f}_tip").id for f in ("index", "middle", "ring", "pinky", "thumb")]
    lo, hi = _limits(model, "right")
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(samples):
        v = rng.uniform(lo, hi)
        v[5] = 0.0
        v[6] = 0.0
        data.qpos[q] = v
        mujoco.mj_kinematics(model, data)
        base = data.xpos[palm_body]
        rotation = data.xmat[palm_body].reshape(3, 3)
        if not (0.12 < base[0] < 0.26 and -0.32 < base[1] < -0.20):
            continue
        if rotation[2, 2] > -0.05 and abs(rotation[2, 2]) > 0.45:  # fingers not pointing up
            continue
        if rotation[1, 0] < 0.7:  # palm faces the midline
            continue
        clearance = min(data.geom_xpos[g, 2] - 0.02 for g in hand_geoms) - TABLE_TOP_Z
        if not (0.10 < clearance < 0.16):
            continue
        if data.site_xpos[tips].max(0)[0] > 0.33:  # fist stays behind the pick region
            continue
        score = abs(rotation[2, 2]) + abs(v[4]) + 0.3 * abs(v[2])
        hits.append((score, v.copy(), base.copy(), rotation[:, 2].copy(), clearance, data.site_xpos[tips].max(0)[0]))
    hits.sort(key=lambda h: h[0])
    print(f"attention: {len(hits)} candidates")
    print(f"{'score':>6} {'joints (7)':<50} {'hand base':<24} {'finger dir':<24} clear  max tip x")
    for score, v, base, fd, clear, tipx in hits[:show]:
        print(f"{score:6.2f} {np.array2string(v, precision=3):<50} {np.array2string(base):<24} {np.array2string(fd):<24} {clear:.3f}  {tipx:.2f}")
    return hits


if __name__ == "__main__":
    which = sys.argv[1:] or ["grasp", "attention"]
    if "grasp" in which:
        sweep_grasp()
    if "attention" in which:
        sweep_attention()
