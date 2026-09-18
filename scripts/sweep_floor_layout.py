"""Find floor basket positions reachable from the default top-grasp lift."""

from __future__ import annotations

import numpy as np

from simulation.pick_place import config as C
from simulation.pick_place.kinematics import rotation_z, solve_pose_ik
from simulation.pick_place.planner import GraspPlanner
from simulation.pick_place.scene import Scene


def sweep() -> list[tuple[float, float, float]]:
    scene = Scene()
    planner = GraspPlanner(scene)
    centers = planner.centers(scene.object_position())
    grasp = solve_pose_ik(scene.model, "right", centers["grasp"], planner.orientation, C.RIGHT_SEED)
    lift = solve_pose_ik(scene.model, "right", centers["lift"], planner.orientation, grasp)
    fingers, thumb = planner.jaw_offsets(planner.orientation)
    jaw = 0.5 * (fingers + thumb)
    jaw_axis = (thumb - fingers) / np.linalg.norm(thumb - fingers)
    _, height = scene.object_extents()
    found = []
    for x in np.linspace(0.10, 0.34, 5):
        for y in np.linspace(-0.52, -0.10, 6):
            for yaw in (0.0, -90.0, 90.0, 180.0):
                turn = rotation_z(yaw)
                place = (
                    np.array([x, y, 0.005 + 0.5 * height + C.PLACE_DROP_HEIGHT])
                    - turn @ jaw
                    + C.JAW_AXIS_BIAS * (turn @ jaw_axis)
                )
                transfer = place.copy()
                transfer[2] = centers["lift"][2]
                try:
                    transfer_joints = solve_pose_ik(
                        scene.model, "right", transfer, turn @ planner.orientation, lift, max_iterations=500
                    )
                    solve_pose_ik(
                        scene.model, "right", place, turn @ planner.orientation, transfer_joints, max_iterations=500
                    )
                except RuntimeError:
                    continue
                found.append((float(x), float(y), float(yaw)))
    print(f"reachable basket poses: {len(found)}")
    for pose in found:
        print(f"  x={pose[0]:.3f} y={pose[1]:.3f} yaw={pose[2]:+.0f}")
    return found


def sweep_pick() -> list[tuple[float, float]]:
    scene = Scene()
    planner = GraspPlanner(scene)
    base = planner.centers(scene.object_position())["grasp"]
    found = []
    for x in np.linspace(0.08, 0.30, 12):
        for y in np.linspace(-0.50, -0.08, 15):
            target = base + [x - scene.pick_position[0], y - scene.pick_position[1], 0.0]
            try:
                solve_pose_ik(
                    scene.model, "right", target, planner.orientation, C.RIGHT_SEED, max_iterations=500
                )
            except RuntimeError:
                continue
            found.append((float(x), float(y)))
    print(f"reachable pick poses: {len(found)}")
    for pose in found:
        print(f"  x={pose[0]:.3f} y={pose[1]:.3f}")
    return found


if __name__ == "__main__":
    sweep_pick()
    sweep()
