"""Log joint states of a pick/place run to CSV for PlotJuggler.

Answers "is the jerkiness from the machine or from the commands?":
  * cmd_* vs pos_*      -- the commanded joint targets and where the joints really are.
                           Steps/kinks in cmd_* are command-side; pos_* lagging a smooth
                           cmd_* is the servo; both smooth = the motion itself is smooth.
  * wall_dt_ms          -- real (wall-clock) time spent per logged sample. Spikes while
                           sim_time advances evenly = the machine/rendering stalled.
  * thinking            -- 1 while the planner is computing (physics paused).
  * phase               -- pick/place phase index (1..7) and which leg (place/retrieve).

    python -m scripts.log_joint_states                     # main task, headless
    python -m scripts.log_joint_states --viewer            # same, with the MuJoCo window
    -> artifacts/joint_logs/<name>.csv  (open in PlotJuggler: File > Load Data,
       choose "sim_time" as the time axis)
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import mujoco
import numpy as np

from simulation.pick_place import executor as executor_module
from simulation.pick_place import episode as episode_module
from simulation.pick_place.retrieve import RetrieveDemo, run_retrieve_trial

OUT_DIR = Path("artifacts") / "joint_logs"


class JointLogger:
    def __init__(self, model: mujoco.MjModel, every: int) -> None:
        self.model, self.every = model, every
        self.rows: list[list] = []
        self.step = 0
        self.last_wall = time.perf_counter()
        self.start_wall = self.last_wall
        self.thinking = 0
        self.phase = 0
        self.leg = 0
        self.joints = {
            side: [model.joint(f"openarm_{side}_joint{i}") for i in range(1, 8)] for side in ("right", "left")
        }
        self.header = ["sim_time", "wall_time", "wall_dt_ms", "thinking", "leg", "phase"]
        for side in ("right", "left"):
            for i in range(1, 8):
                self.header += [f"{side}/j{i}/cmd", f"{side}/j{i}/pos", f"{side}/j{i}/vel",
                                f"{side}/j{i}/err_deg", f"{side}/j{i}/torque"]
        self.header += ["object/z", "object/tilt_deg"]

    def sample(self, data: mujoco.MjData, arm_actuators: dict, force: bool = False) -> None:
        self.step += 1
        if not force and self.step % self.every:
            return
        now = time.perf_counter()
        row = [float(data.time), now - self.start_wall, (now - self.last_wall) * 1000.0, self.thinking, self.leg, self.phase]
        self.last_wall = now
        for side in ("right", "left"):
            ctrl = data.ctrl[arm_actuators[side]]
            for i, joint in enumerate(self.joints[side]):
                q = float(data.qpos[joint.qposadr[0]])
                row += [float(ctrl[i]), q, float(data.qvel[joint.dofadr[0]]),
                        float(np.degrees(ctrl[i] - q)), float(data.actuator_force[arm_actuators[side][i]])]
        body = self.model.body("pick_bottle").id
        z_axis = data.xmat[body].reshape(3, 3)[:, 2]
        row += [float(data.xpos[body][2]), float(np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0))))]
        self.rows.append(row)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.header)
            writer.writerows(self.rows)


def install(logger: JointLogger) -> None:
    """Hook the executor's physics step, its planner calls and the phase log."""
    original_step = executor_module.Executor._step

    def step(self) -> None:
        original_step(self)
        logger.sample(self.data, self.scene.arm_actuators)

    executor_module.Executor._step = step

    original_think = executor_module.Executor.think

    def think(self, fn, *args, **kwargs):
        logger.thinking = 1
        logger.sample(self.data, self.scene.arm_actuators, force=True)
        try:
            return original_think(self, fn, *args, **kwargs)
        finally:
            logger.sample(self.data, self.scene.arm_actuators, force=True)
            logger.thinking = 0

    executor_module.Executor.think = think

    original_phase = episode_module.EpisodeLog.phase

    def phase(self, index, total, name, *rest, **kwargs):
        if index == 1:
            logger.leg += 1
        logger.phase = index
        return original_phase(self, index, total, name, *rest, **kwargs)

    episode_module.EpisodeLog.phase = phase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pick", type=float, nargs=2, default=[0.28, -0.25])
    parser.add_argument("--basket", type=float, nargs=2, default=[0.28, 0.0])
    parser.add_argument("--retrieve-to", type=float, nargs=2, default=[0.28, 0.25])
    parser.add_argument("--platform", type=float, default=0.10)
    parser.add_argument("--every", type=int, default=2, help="log every Nth physics step (1ms each)")
    parser.add_argument("--viewer", action="store_true", help="run with the MuJoCo window (includes render pacing)")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    task = RetrieveDemo(tuple(args.pick), tuple(args.basket), tuple(args.retrieve_to),
                        side="left", place_side="right", work_platform_height=args.platform)
    logger = JointLogger(task.place_in.model, args.every)
    install(logger)
    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(task.place_in.model, task.place_in.data) as viewer:
            result = run_retrieve_trial(task, viewer)
    else:
        result = run_retrieve_trial(task)
    name = args.name or f"retrieve_{'viewer' if args.viewer else 'headless'}_{time.strftime('%Y%m%d_%H%M%S')}"
    path = OUT_DIR / f"{name}.csv"
    logger.write(path)
    print(f"success={result.success} rows={len(logger.rows)} -> {path}")


if __name__ == "__main__":
    main()
