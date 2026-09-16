from __future__ import annotations

import argparse
import time

import mujoco.viewer

from simulation.openarm_mujoco import MujocoRobot


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the official OpenArm v2 bottle scene")
    parser.add_argument("--headless", action="store_true", help="Load and step without opening a viewer")
    parser.add_argument("--seconds", type=float, default=2.0, help="Headless simulation duration")
    parser.add_argument("--stock-gripper", action="store_true", help="Use the stock two-finger gripper")
    args = parser.parse_args()

    robot = MujocoRobot(five_finger=not args.stock_gripper)
    robot.reset()
    if args.headless:
        robot.step(args.seconds)
        pose = robot.end_effector_pose()
        print(f"OpenArm MuJoCo OK: nq={robot.model.nq}, nv={robot.model.nv}, nu={robot.model.nu}, left_ee={pose.position}")
        return

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        while viewer.is_running():
            started = time.monotonic()
            mujoco.mj_step(robot.model, robot.data)
            viewer.sync()
            time.sleep(max(0.0, robot.model.opt.timestep - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
