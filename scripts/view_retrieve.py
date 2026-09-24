"""Watch the basket retrieve task in the MuJoCo window.

One arm picks the can and puts it in the basket (optionally on a stand), then grasps
it back out and sets it down on the bare table; `--retrieve-arm left` makes the other
arm do the retrieval (no robust layout exists for that on the real-height robot --
see context_project.md). The executor aborts
the run on any arm/hand contact with the table, basket, stand or robot body, on the
two arms touching, and on the idle arm touching the can -- a finished run means none
of that happened.

    python -m scripts.view_retrieve                      # the committed layout
    python -m scripts.view_retrieve --place-arm right --retrieve-arm left --pick 0.20 -0.30         --basket 0.25 0.0 --retrieve-to 0.15 0.35 --floor-tilt 12   # centre basket, V-floor
    python -m scripts.view_retrieve --place-arm right --retrieve-arm left --pick 0.28 -0.25         --basket 0.28 0.0 --retrieve-to 0.28 0.25 --platform 0.10      # centre basket, 10cm platform

Keys in the window: R = replay the recorded run (smooth, no planning pauses; press
again to restart it), Esc = free camera, [ / ] = cameras, close the window to quit.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from simulation.pick_place.retrieve import RetrieveDemo, run_retrieve_trial

# The layout tests/test_retrieve.py checks (test_place_then_retrieve_from_basket).
DEFAULT_PICK = (0.08, -0.38)
DEFAULT_BASKET = (0.25, -0.25)
DEFAULT_RETRIEVE_TO = (0.15, -0.40)
DEFAULT_STAND = 0.0


def build_task(args: argparse.Namespace) -> RetrieveDemo:
    return RetrieveDemo(
        tuple(args.pick), tuple(args.basket), tuple(args.retrieve_to), side=args.retrieve_arm, place_side=args.place_arm,
        basket_stand_height=args.stand, basket_floor_tilt_deg=args.floor_tilt, work_platform_height=args.platform, place_offset=tuple(args.place_offset) if args.place_offset else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pick", type=float, nargs=2, default=list(DEFAULT_PICK))
    parser.add_argument("--basket", type=float, nargs=2, default=list(DEFAULT_BASKET))
    parser.add_argument("--retrieve-to", type=float, nargs=2, default=list(DEFAULT_RETRIEVE_TO))
    parser.add_argument("--place-arm", choices=("right", "left"), default="right")
    parser.add_argument("--retrieve-arm", choices=("right", "left"), default="right")
    parser.add_argument("--platform", type=float, default=0.0, help="work platform height (m) under object/basket/set-down")
    parser.add_argument("--floor-tilt", type=float, default=0.0, help="V-floor insert slope (deg), 0 = flat floor")
    parser.add_argument("--stand", type=float, default=DEFAULT_STAND, help="basket stand height (m), 0 = on the table")
    parser.add_argument("--place-offset", type=float, nargs=2, default=None, help="release point inside the basket (m)")
    args = parser.parse_args()

    task = build_task(args)
    replay_requested = [False]

    def on_key(keycode: int) -> None:
        if keycode == ord("R"):
            replay_requested[0] = True

    with mujoco.viewer.launch_passive(task.place_in.model, task.place_in.data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.20, 0.0, 0.25]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 200
        viewer.cam.elevation = -25
        recorder = RecordingViewer(viewer, task.place_in.model, task.place_in.data)
        _status(viewer, "running (planning pauses are the robot thinking)")
        print(f"=== {args.place_arm} hand places into the basket, {args.retrieve_arm} hand takes it out ===")
        result = run_retrieve_trial(task, recorder)
        print(f"  success: {result.success}")
        if result.failure_reason:
            print(f"  failure: {result.failure_reason}")
        print(
            f"  final can {[round(v, 3) for v in result.final_position]}, "
            f"set-down error {result.placement_error_m*1000:.1f}mm, tilt {result.bottle_tilt_deg:.1f}deg, "
            f"max penetration {result.max_penetration_m*1000:.1f}mm"
        )
        done = "success" if result.success else "FAILED"
        print(f"  recorded {len(recorder.frames)} frames; press R in the window to replay, close it to quit")
        _status(viewer, f"{done} -- press R to replay")
        replay_requested[0] = False
        while viewer.is_running():
            if replay_requested[0]:
                replay_requested[0] = False
                _status(viewer, "replaying (R = restart replay)")
                recorder.replay(replay_requested)
                _status(viewer, f"{done} -- press R to replay")
            viewer.sync()
            time.sleep(0.02)


class RecordingViewer:
    """Pass-through to the MuJoCo viewer that keeps one qpos snapshot per rendered
    frame, so the run can be replayed afterwards at real speed without re-planning."""

    def __init__(self, viewer, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._viewer, self._model, self._data = viewer, model, data
        self.frames: list[tuple[float, np.ndarray]] = []

    def __getattr__(self, name):
        return getattr(self._viewer, name)

    def sync(self) -> None:
        t = float(self._data.time)
        if not self.frames or t > self.frames[-1][0]:
            self.frames.append((t, self._data.qpos.copy()))
        self._viewer.sync()

    def replay(self, interrupt: list[bool]) -> None:
        model, data, viewer = self._model, self._data, self._viewer
        start_wall, start_sim = time.perf_counter(), self.frames[0][0] if self.frames else 0.0
        final_qpos, final_time = data.qpos.copy(), float(data.time)
        for t, qpos in self.frames:
            if not viewer.is_running() or interrupt[0]:
                break
            with viewer.lock():
                data.qpos[:] = qpos
                data.time = t
                mujoco.mj_kinematics(model, data)
            viewer.sync()
            ahead = (t - start_sim) - (time.perf_counter() - start_wall)
            if ahead > 0.0:
                time.sleep(ahead)
        if interrupt[0]:
            return
        with viewer.lock():
            data.qpos[:] = final_qpos
            data.time = final_time
            mujoco.mj_kinematics(model, data)


def _status(viewer, text: str) -> None:
    try:
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Status", text))
    except (AttributeError, TypeError):
        pass


if __name__ == "__main__":
    main()
