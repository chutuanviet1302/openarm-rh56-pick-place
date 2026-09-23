"""Sweep the table with the real planner and export IK-verified reach polygons.

Output: viewer/reach_map.json -- for each arm, a boolean grid of "can grasp a can
standing here" and "can set a can down here (bare table)", each cell probed the way
plan_pick / plan_place probe them (same wrist targets, every grasp/place yaw, both
IK seeds, joint margin >= MIN_JOINT_MARGIN_DEG). control.js draws these grids as
convex hulls, so the browser-side validity check agrees with what TaskRouter's
planners will actually solve.

    python -m scripts.sweep_reach_map            # ~2-4 min, writes the JSON
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from simulation.pick_place import config as C
from simulation.pick_place.kinematics import rotation_z, solve_pose_ik
from simulation.pick_place.planner import GraspPlanner
from simulation.pick_place.scene import Scene

OUT_PATH = Path("viewer") / "reach_map.json"
X_STEP = Y_STEP = 0.05
X_RANGE = (-0.15, 0.60)
Y_RANGE = (-0.50, 0.50)
PROBE_ITERS = 350


def main() -> None:
    xs = [round(float(x), 3) for x in np.arange(X_RANGE[0], X_RANGE[1] + 1e-9, X_STEP)]
    ys = [round(float(y), 3) for y in np.arange(Y_RANGE[0], Y_RANGE[1] + 1e-9, Y_STEP)]
    grid = {
        side: {"grasp": [[False] * len(ys) for _ in xs], "place": [[False] * len(ys) for _ in xs]}
        for side in ("right", "left")
    }
    started = time.perf_counter()
    for side in ("right", "left"):
        scene = Scene((0.08, -0.38), (0.25, -0.25))
        planner = GraspPlanner(scene, side)
        can_z = float(scene.object_position()[2])
        _, height = scene.object_extents()
        seeds = [C.ARM_SEED[side]]
        seeds.append(C.NATURAL_GRASP_JOINTS if side == "right" else C.NATURAL_GRASP_JOINTS * C.MIRROR_JOINT_SIGNS)

        def solves(target: np.ndarray, yaw: float, seed: np.ndarray) -> bool:
            try:
                q = solve_pose_ik(scene.model, side, target, rotation_z(yaw) @ planner.base_orientation, seed, max_iterations=PROBE_ITERS)
            except RuntimeError:
                return False
            return planner.joint_margin_degrees(q) >= C.MIN_JOINT_MARGIN_DEG

        # A probe reference position: `centers()` needs an object pose; only its XY
        # offsets matter for the grasp target's height, which comes from the can.
        probe_object = np.array([0.20, -0.25, can_z])
        for ix, x in enumerate(xs):
            here = np.array([x, 0.0, can_z])
            for iy, y in enumerate(ys):
                object_xy = here + [0.0, y, 0.0]
                # --- grasp probe: the planner's own grasp wrist target for a can here
                try:
                    centers = planner.centers(object_xy)
                except RuntimeError:
                    centers = None
                if centers is not None:
                    for yaw in C.GRASP_YAW_CANDIDATES_DEG:
                        if any(solves(centers["grasp"], yaw, seed) for seed in seeds):
                            grid[side]["grasp"][ix][iy] = True
                            break
                # --- place probe: the planner's own set-down wrist target for a bare-table
                # can drop here (object reference XY only biases the yaw offsets, and the
                # place target depends on the floor + hand geometry, not the pick XY)
                done = False
                place_floor = np.array([x, y, 0.005])
                for yaw in C.PLACE_YAW_CANDIDATES_DEG:
                    try:
                        cy = planner.centers(probe_object, yaw, None, place_floor)
                    except RuntimeError:
                        continue
                    if any(solves(cy["lower"], yaw, seed) for seed in seeds):
                        grid[side]["place"][ix][iy] = True
                        done = True
                        break
            print(f"{side} x={x:+.2f} done ({time.perf_counter() - started:.0f}s)", flush=True)

    payload = {"xs": xs, "ys": ys, "step": [X_STEP, Y_STEP], **grid}
    OUT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    counts = {side: {k: sum(map(sum, v)) for k, v in body.items()} for side, body in grid.items()}
    print("cells reachable:", counts)
    print(f"wrote {OUT_PATH} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
