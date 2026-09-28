"""Both arms clear their side of the table into one box: each arm finds, identifies and
drops two different objects, in different rest poses, into the box between them.

    python -m simulation.pick_place.bin_task                  # simulate, then play it back in MuJoCo
    python -m simulation.pick_place.bin_task --no-view        # simulate + save only
    python -m simulation.pick_place.bin_task --replay artifacts/bin_task_frames.npz

Smooth viewing: the task is simulated off-screen first (planning never stalls the
picture), every 1/60 s of sim time the pose of the scene is recorded, then the
recording is played back in the MuJoCo viewer at real speed (qpos + kinematics only).
The recording is saved, so it can be watched again with --replay.

Per arm (right first, then left), twice:
    look      head camera RGB-D -> ObjectDetector (several objects of one kind allowed;
              one reference per kind and rest pose) -> ObjectTracker
    choose    an object on this arm's side of the table, nearest the box
    pick      pose backend -> grasp library (upright / lying grasp from the pose) ->
              plan -> grasp -> proof lift -> carry over the box -> let go just above
              the rim over this arm's half of the box (walls 8 cm: nothing bounces out)
    verify    look again (gone from the table) and the box check (inside the walls,
              below the rim)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from simulation.object_detector import ObjectDetector, ObjectTracker
from simulation.objects import Placement
from simulation.pick_place import config as C
from simulation.pick_place.demo import Demo, run_trial
from simulation.pick_place.scene import Scene

PLATFORM = 0.10
BOX = (0.28, 0.0)
BOX_HALF = (0.11, 0.14)       # inner half size x, y: 22 x 28 cm
BOX_WALL = 0.08
# Right side (y < 0) for the right arm, left side (y > 0) for the left arm; two
# different objects each, in different rest poses. Every spot was plan-checked first
# (drop over the box, 2026-09-28): the right arm takes the lying can and a peach; the
# left arm the apple and the orange -- it found no grasp for a can (upright or lying)
# at (0.28, +0.27) within 400 s of planning. (A flat tuna can at (0.38, -0.30) tipped
# 78 deg on the proof lift and ended on its side: the fingers held its 34 mm rim.)
LAYOUT = (
    Placement("can", (0.25, -0.22), "lying", 30.0),
    Placement("peach", (0.38, -0.30), name="peach"),
    Placement("apple", (0.28, 0.26), name="apple"),
    Placement("orange", (0.38, 0.30), name="orange"),
)
KNOWN = (("can", "upright"), ("can", "lying"), ("peach", "upright"), ("orange", "upright"), ("apple", "upright"))
PICKS_PER_ARM = 2
MAX_ATTEMPTS = 2
# Where each arm lets go over the box (offsets from its centre): its own half, one
# spot per pick.
DROP_SPOTS = {"right": [(-0.04, -0.06), (0.0, -0.06)], "left": [(-0.04, 0.06), (0.04, 0.06)]}
FRAME_SECONDS = 1.0 / 60.0


def build_scene(layout=LAYOUT) -> Scene:
    layout = list(layout)
    first = layout[0]
    return Scene(first.xy, BOX, work_platform_height=PLATFORM, pick_object=first.key, pick_pose=first.pose,
                 pick_yaw_deg=first.yaw_deg, extra_objects=layout[1:], basket_half_size=BOX_HALF,
                 basket_wall_height=BOX_WALL)


class FrameRecorder:
    """Stands in for the viewer while simulating off-screen: the executor calls sync()
    once per `frame_seconds` of sim time and, `realtime` being False, neither paces to
    the wall clock nor runs the planner in a thread (scripts/view_retrieve.py)."""

    realtime = False

    def __init__(self, data: mujoco.MjData, frame_seconds: float = FRAME_SECONDS) -> None:
        self._data = data
        self.frame_seconds = frame_seconds
        self.times: list[float] = []
        self.qpos: list[np.ndarray] = []
        self.captions: list[str] = []
        self.caption = ""

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        t = float(self._data.time)
        if not self.times or t > self.times[-1]:
            self.times.append(t)
            self.qpos.append(self._data.qpos.copy())
            self.captions.append(self.caption)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, times=np.asarray(self.times), qpos=np.asarray(self.qpos),
                            captions=np.asarray(self.captions))


@dataclass
class PickRecord:
    arm: str
    object: str
    kind: str
    attempt: int
    success: bool
    grasp: str | None
    failure: str | None
    failed_phase: str | None
    proof_lift_slip_mm: float | None
    final_xyz: list[float]


@dataclass
class BinResult:
    pose_backend: str
    picks: list[PickRecord] = field(default_factory=list)
    in_box: dict[str, bool] = field(default_factory=dict)
    per_arm: dict[str, list[str]] = field(default_factory=lambda: {"right": [], "left": []})
    sim_seconds: float = 0.0
    wall_seconds: float = 0.0

    def summary(self) -> str:
        return (f"right arm: {self.per_arm['right']}, left arm: {self.per_arm['left']}; "
                f"in the box {sum(self.in_box.values())}/{len(self.in_box)} {self.in_box}; "
                f"sim {self.sim_seconds:.0f}s, wall {self.wall_seconds:.0f}s")


class BinTask:
    # Subclasses (bin_conveyor_task) swap these.
    box, box_half, box_wall, known, drop_spots = BOX, BOX_HALF, BOX_WALL, KNOWN, DROP_SPOTS

    def __init__(self, layout=LAYOUT, *, pose_backend: str = "gt") -> None:
        self.layout = list(layout)
        self.scene = self.build_scene()
        self.pose_backend = pose_backend
        self.detector = ObjectDetector(self.scene.model, self.known, basket_xy=self.box,
                                       basket_margin=max(self.box_half) + 0.03)
        self.tracker = ObjectTracker()
        self.recorder = FrameRecorder(self.scene.data)

    def build_scene(self) -> Scene:
        return build_scene(self.layout)

    def enroll(self) -> None:
        def alone(key: str, yaw: float, pose: str = "upright") -> Scene:
            return Scene((0.28, -0.27), self.box, work_platform_height=PLATFORM, pick_object=key, pick_pose=pose,
                         pick_yaw_deg=yaw, basket_half_size=self.box_half, basket_wall_height=self.box_wall)

        self.detector.enroll(alone)

    def look(self) -> list:
        detections = self.detector.detect(self.scene.data, unique=False)
        self.tracker.update(detections, float(self.scene.data.time))
        unknown = [d for d in detections if d.label is None]
        if unknown:
            print("   unidentified: " + "; ".join(
                f"cost {d.cost:.0f} at {np.round(d.centroid[:2], 3).tolist()} {d.pixels}px {d.features.as_dict()}"
                for d in unknown))
        return [d for d in detections if d.label is not None]

    def mask_source(self, name: str):
        """A fresh detection's pixels for the object `name` (nearest detection of its
        kind), for every pose estimate of a pick -- retries included."""
        def fresh() -> np.ndarray | None:
            kind = self.scene.object_types[name]
            detections = [d for d in self.detector.detect(self.scene.data, unique=False) if d.label == kind]
            if not detections:
                raise RuntimeError(f"perception failed: {name} not seen any more")
            here = self.scene.object_position_of(name)[:2]
            return min(detections, key=lambda d: float(np.linalg.norm(d.centroid[:2] - here))).mask
        return fresh

    def instance_of(self, detection) -> str:
        """The simulator's name for the detected object (bookkeeping for the gt pose
        backend and the report): the instance of that kind nearest the detection."""
        scene = self.scene
        candidates = [name for name, kind in scene.object_types.items() if kind == detection.label]
        return min(candidates, key=lambda n: float(np.linalg.norm(scene.object_position_of(n)[:2] - detection.centroid[:2])))

    def run(self) -> BinResult:
        started = time.perf_counter()
        result = BinResult(self.pose_backend)
        scene = self.scene
        print("enrolling the objects (one reference per kind and pose) ...")
        self.enroll()
        attempts: dict[str, int] = {}
        for arm in ("right", "left"):
            spots = list(self.drop_spots[arm])
            empty_looks = 0
            while len(result.per_arm[arm]) < PICKS_PER_ARM:
                detections = self.look()
                mine = [d for d in detections if (d.centroid[1] < 0) == (arm == "right")]
                mine = [d for d in mine if attempts.get(self.instance_of(d), 0) < MAX_ATTEMPTS]
                seen = ", ".join(f"{d.label} at {np.round(d.centroid[:2], 3).tolist()}" for d in detections)
                print(f"[look t={scene.data.time:6.1f}s] {seen or 'nothing on the table'}")
                if not mine and empty_looks < 2 and any(
                        (t.position[1] < 0) == (arm == "right") and t.missed <= 1 for t in self.tracker.tracks.values()):
                    # Objects seen on this side a moment ago: look again before giving up.
                    empty_looks += 1
                    self._settle(0.5)
                    continue
                if not mine:
                    print(f"   nothing left for the {arm} arm")
                    break
                empty_looks = 0
                detection = min(mine, key=lambda d: float(np.hypot(d.centroid[0] - self.box[0], d.centroid[1] - self.box[1])))
                name = self.instance_of(detection)
                attempts[name] = attempts.get(name, 0) + 1
                spot = spots[0] if spots else (0.0, -0.06 if arm == "right" else 0.06)
                print(f"== {arm} arm -> {detection.label} ({name}), attempt {attempts[name]}, drop over box spot {spot}")
                self.recorder.caption = f"{arm} arm -> {detection.label} ({name})"
                scene.set_target(name)
                demo = Demo(scene=scene, side=arm, pose_backend=self.pose_backend, release="drop",
                            place_offset=spot, detection_mask=detection.mask, mask_source=self.mask_source(name))
                trial = run_trial(demo, self.recorder)
                self._settle(1.0)
                ok = scene.object_in_basket(name)
                slip = None
                if trial.proof_lift_hand_rise_m is not None and trial.proof_lift_rise_m is not None:
                    slip = round((trial.proof_lift_hand_rise_m - trial.proof_lift_rise_m) * 1000, 1)
                result.picks.append(PickRecord(
                    arm, name, scene.object_types[name], attempts[name], ok, trial.grasp_name,
                    None if ok else (trial.failure_reason or "not in the box").splitlines()[0][:200],
                    trial.failed_phase, slip, np.round(scene.object_position_of(name), 4).tolist(),
                ))
                print(f"   -> {'IN THE BOX' if ok else 'FAILED: ' + str(result.picks[-1].failure)}")
                self.recorder.caption = f"{arm} arm: {name} {'in the box' if ok else 'missed'}"
                if ok:
                    result.per_arm[arm].append(name)
                    if spots:
                        spots.pop(0)
                self._home(arm)
        self.after_table(result)
        result.in_box = {name: scene.object_in_basket(name) for name in scene.object_types}
        result.sim_seconds = float(scene.data.time)
        result.wall_seconds = time.perf_counter() - started
        return result

    def after_table(self, result: BinResult) -> None:
        """Hook after both arms have cleared the table (bin_conveyor_task: the belt)."""

    def _settle(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.scene.model.opt.timestep))):
            mujoco.mj_step(self.scene.model, self.scene.data)
            if float(self.scene.data.time) - (self.recorder.times[-1] if self.recorder.times else -1.0) >= FRAME_SECONDS:
                self.recorder.sync()

    def _home(self, side: str) -> None:
        """Back to attention if a failed pick left the arm elsewhere."""
        from simulation.pick_place.executor import Executor

        scene = self.scene
        if np.max(np.abs(scene.data.qpos[scene.arm_qpos[side]] - scene.attention_pose[side])) < 0.05:
            return
        executor = Executor(scene)
        executor.viewer = self.recorder
        executor.active_side = side
        try:
            executor.move_to({f"{side}_hand": scene.hand_ctrl(side, open_fingers=C.ALL_FINGERS)}, C.RELEASE_SECONDS)
            # Straight up first: a failed pick can leave the hand low over the table or
            # belt, and a direct blend home swept a finger into the belt.
            from simulation.pick_place.kinematics import solve_pose_ik, wrist_frame

            here = scene.data.ctrl[scene.arm_actuators[side]].copy()
            position, rotation = wrist_frame(scene.model, side, here)
            try:
                up = solve_pose_ik(scene.model, side, position + np.array([0.0, 0.0, 0.12]), rotation, here)
                executor.move_to({f"{side}_arm": up}, C.RETURN_SECONDS)
            except RuntimeError:
                pass

            executor.move_to({f"{side}_arm": scene.attention_pose[side], f"{side}_hand": scene.rest_hand[side]},
                             2.0 * C.RETURN_SECONDS)
        except RuntimeError as error:
            print(f"   (homing the {side} arm stopped: {str(error).splitlines()[0]})")


def replay(path: Path, speed: float = 1.0, scene_builder=build_scene) -> None:
    """Play a saved recording in the MuJoCo viewer at real speed, then keep the last
    frame up until the window is closed."""
    import mujoco.viewer

    recording = np.load(path)
    times, qpos, captions = recording["times"], recording["qpos"], recording["captions"]
    scene = scene_builder()
    model, data = scene.model, scene.data
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.30, 0.0, 0.20]
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 180.0, -35.0, 1.45
        start_wall, start_sim, caption = time.perf_counter(), float(times[0]), None
        for t, q, text in zip(times, qpos, captions):
            if not viewer.is_running():
                return
            with viewer.lock():
                data.qpos[:] = q
                data.time = float(t)
                mujoco.mj_kinematics(model, data)
                if text != caption:
                    caption = text
                    try:
                        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                          f"t = {t:5.1f} s", str(text)))
                    except (AttributeError, TypeError):
                        pass
            viewer.sync()
            ahead = (float(t) - start_sim) / speed - (time.perf_counter() - start_wall)
            if ahead > 0.0:
                time.sleep(ahead)
        print("playback finished; close the viewer window to exit")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Both arms drop two objects each into a box")
    parser.add_argument("--pose-backend", default="gt", choices=("gt", "foundationpose"))
    parser.add_argument("--frames", type=Path, default=Path("artifacts") / "bin_task_frames.npz")
    parser.add_argument("--report", type=Path, default=Path("artifacts") / "bin_task.json")
    parser.add_argument("--no-view", action="store_true", help="simulate and save only")
    parser.add_argument("--replay", type=Path, help="play a saved recording and exit")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    args = parser.parse_args(argv)
    if args.replay:
        replay(args.replay, args.speed)
        return
    task = BinTask(pose_backend=args.pose_backend)
    try:
        result = task.run()
    finally:
        task.recorder.save(args.frames)  # keep the recording even if the run crashed
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(asdict(result), indent=2, default=float), encoding="utf-8")
    print("\n" + result.summary())
    print(f"recording: {args.frames} ({len(task.recorder.times)} frames); report: {args.report}")
    if not args.no_view:
        replay(args.frames, args.speed)


if __name__ == "__main__":
    main()
