"""Probe whether both RH56 hands can reach one object without inter-hand overlap.

Run: python -m scripts.check_handoff_geometry
This is a static pose screen, not a proof of an executable handoff trajectory.
"""

from __future__ import annotations

import mujoco
import numpy as np

from simulation.pick_place import config as C
from simulation.pick_place.kinematics import rotation_z, solve_pose_ik
from simulation.pick_place.planner import GraspPlanner
from simulation.pick_place.scene import Scene


def main() -> None:
    scene = Scene((0.08, -0.38), (0.25, 0.25))
    model = scene.model
    planners = {side: GraspPlanner(scene, side) for side in ("right", "left")}
    best: tuple[float, int, tuple] | None = None
    pairs = 0
    for x in (0.25, 0.30, 0.35, 0.40, 0.45):
        for z in (0.20, 0.30, 0.40, 0.50):
            candidates: dict[str, list[tuple[int, float, np.ndarray]]] = {}
            for side, yaws in (("right", (45, 90, 135, 180)), ("left", (-180, -135, -90, -45))):
                planner = planners[side]
                candidates[side] = []
                for yaw in yaws:
                    orientation = rotation_z(yaw) @ planner.base_orientation
                    fingers, thumb = planner.jaw_offsets(orientation)
                    for dz in (-0.06, -0.03, 0.0, 0.03, 0.06):
                        wrist = np.array([x, 0.0, z]) - 0.5 * (fingers + thumb) + [0, 0, dz]
                        try:
                            joints = solve_pose_ik(model, side, wrist, orientation, C.ARM_SEED[side])
                        except RuntimeError:
                            continue
                        if planner.joint_margin_degrees(joints) >= C.MIN_JOINT_MARGIN_DEG:
                            candidates[side].append((yaw, dz, joints))
            for right_yaw, right_dz, right_joints in candidates["right"]:
                for left_yaw, left_dz, left_joints in candidates["left"]:
                    data = mujoco.MjData(model)
                    for side, joints, closed in (
                        ("right", right_joints, True),
                        ("left", left_joints, False),
                    ):
                        hand = planners[side]._pregrasp_data(joints, closed=closed)
                        data.qpos[scene.arm_qpos[side]] = joints
                        data.qpos[scene.hand_qpos[side]] = hand.qpos[scene.hand_qpos[side]]
                    mujoco.mj_forward(model, data)
                    overlaps = [
                        -contact.dist
                        for contact in data.contact[: data.ncon]
                        if {scene.hand_side(contact.geom1), scene.hand_side(contact.geom2)} == {"left", "right"}
                        and contact.dist < 0
                    ]
                    pairs += 1
                    candidate = (max(overlaps, default=0.0), len(overlaps), (x, z, right_yaw, right_dz, left_yaw, left_dz))
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate
    print(f"IK/joint-margin-valid hand pairs: {pairs}")
    if best is None:
        print("No common IK pose in sampled workspace")
    else:
        depth, contacts, pose = best
        print(f"Lowest maximum inter-hand penetration: {depth * 1000:.1f} mm, {contacts} contacts; pose={pose}")
        print("Collision-free handoff candidate found" if depth <= 0 else "No collision-free candidate in sampled workspace")


if __name__ == "__main__":
    main()
