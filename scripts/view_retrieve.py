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

By default the whole task is simulated first (no window) and then played back at
real speed, so the window runs smoothly -- no stalls while the planner thinks.
`--live` watches the simulation as it runs instead (planning pauses included);
`--speed 2` plays back twice as fast.

Keys in the window: R = replay from the start, Esc = free camera, [ / ] = cameras,
close the window to quit.
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
    parser.add_argument("--live", action="store_true", help="simulate while watching (old mode: planning pauses)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed, e.g. 2 = twice real time")
    parser.add_argument("--fps", type=float, default=60.0, help="recorded frames per second of robot time")
    args = parser.parse_args()

    task = build_task(args)
    if args.live:
        run_live(task, args)
        return

    # Default: simulate the whole task first (no window, no real-time pacing), then
    # play it back. Watching it live stalls for every planner call (~60s in total on
    # the centre-basket task) and physics shares the CPU with rendering; the playback
    # only poses the recorded frames, so it runs smoothly at real speed.
    print(f"=== {args.place_arm} hand places into the basket, {args.retrieve_arm} hand takes it out ===")
    print("simulating the whole task first (planning included), then playing it back ...")
    recorder = FrameRecorder(task.place_in.model, task.place_in.data, 1.0 / args.fps)
    started = time.perf_counter()
    result = run_retrieve_trial(task, recorder)
    _print_result(result)
    print(f"  simulated in {time.perf_counter() - started:.0f}s; {len(recorder.frames)} frames "
          f"({recorder.frames[-1][0]:.1f}s of robot time) -- opening the window")
    done = "success" if result.success else "FAILED"
    replay_requested = [False]

    def on_key(keycode: int) -> None:
        if keycode == ord("R"):
            replay_requested[0] = True

    with mujoco.viewer.launch_passive(task.place_in.model, task.place_in.data, key_callback=on_key) as viewer:
        _camera(viewer)
        player = RecordingViewer(viewer, task.place_in.model, task.place_in.data, recorder.frames)
        while viewer.is_running():
            _status(viewer, f"playing x{args.speed:g} (R = restart)")
            player.replay(replay_requested, args.speed)
            _status(viewer, f"{done} -- press R to replay")
            replay_requested[0] = False
            while viewer.is_running() and not replay_requested[0]:
                viewer.sync()
                time.sleep(0.02)
            replay_requested[0] = False


def run_live(task: RetrieveDemo, args: argparse.Namespace) -> None:
    """Simulate while watching (planning pauses and all); R replays afterwards."""
    replay_requested = [False]

    def on_key(keycode: int) -> None:
        if keycode == ord("R"):
            replay_requested[0] = True

    with mujoco.viewer.launch_passive(task.place_in.model, task.place_in.data, key_callback=on_key) as viewer:
        _camera(viewer)
        recorder = RecordingViewer(viewer, task.place_in.model, task.place_in.data)
        _status(viewer, "running (planning pauses are the robot thinking)")
        print(f"=== {args.place_arm} hand places into the basket, {args.retrieve_arm} hand takes it out ===")
        result = run_retrieve_trial(task, recorder)
        _print_result(result)
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


def _camera(viewer) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0.20, 0.0, 0.25]
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 200
    viewer.cam.elevation = -25


def _print_result(result) -> None:
    print(f"  success: {result.success}")
    if result.failure_reason:
        print(f"  failure: {result.failure_reason}")
    print(
        f"  final can {[round(v, 3) for v in result.final_position]}, "
        f"set-down error {result.placement_error_m*1000:.1f}mm, tilt {result.bottle_tilt_deg:.1f}deg, "
        f"max penetration {result.max_penetration_m*1000:.1f}mm"
    )


class FrameRecorder:
    """Stands in for the viewer while the task is simulated off-screen: the executor
    calls sync() once per `frame_seconds` of robot time and, since `realtime` is
    False, neither paces to the wall clock nor runs the planner in a thread."""

    realtime = False

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, frame_seconds: float) -> None:
        self._data = data
        self.frame_seconds = frame_seconds
        self.frames: list[tuple[float, np.ndarray]] = []

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        t = float(self._data.time)
        if not self.frames or t > self.frames[-1][0]:
            self.frames.append((t, self._data.qpos.copy()))


class RecordingViewer:
    """Pass-through to the MuJoCo viewer that keeps one qpos snapshot per rendered
    frame, so the run can be replayed afterwards at real speed without re-planning."""

    def __init__(self, viewer, model: mujoco.MjModel, data: mujoco.MjData, frames=None) -> None:
        self._viewer, self._model, self._data = viewer, model, data
        self.frames: list[tuple[float, np.ndarray]] = frames if frames is not None else []

    def __getattr__(self, name):
        return getattr(self._viewer, name)

    def sync(self) -> None:
        t = float(self._data.time)
        if not self.frames or t > self.frames[-1][0]:
            self.frames.append((t, self._data.qpos.copy()))
        self._viewer.sync()

    def replay(self, interrupt: list[bool], speed: float = 1.0) -> None:
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
            ahead = (t - start_sim) / speed - (time.perf_counter() - start_wall)
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
