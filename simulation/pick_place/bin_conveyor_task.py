"""The bin task, then a conveyor: after both arms have cleared the table into the box,
a belt behind the box starts and brings two more objects; each arm meets one on the
moving belt and drops it into the same box.

    python -m simulation.pick_place.bin_conveyor_task                     # simulate, then play back
    python -m simulation.pick_place.bin_conveyor_task --pose-backend foundationpose
    python -m simulation.pick_place.bin_conveyor_task --replay artifacts/bin_conveyor_frames.npz

Layout (all spots plan-checked before use): the box moves in to (0.24, 0) and narrows
to 18 x 28 cm so the belt (x = 0.40, 8 cm wide, top 1 cm over the platform) runs
behind it; the table objects stay off the belt strip. The belt objects wait upstream
(y = 0.9, 1.05 m), out of the camera's view, while the belt stands still.

Phase 2, on the belt (sim time; the belt never stops once started):
    look      RGB-D on the belt strip -> ObjectDetector -> ObjectTracker (velocity fit)
    assign    an arm with belt picks left (BELT_PICKS; the right arm takes both here: the
              belt runs where only it reaches) if the object can still get to that
              arm's intercept point late enough
    pick      ConveyorDemo (conveyor.py): plan for the object where it will be, wait
              at the standoff, correct from a fresh look, close while riding along with
              the belt, lift off the belt, drop into the box
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from simulation.five_finger_model import Conveyor
from simulation.object_detector import ObjectDetector, ObjectTracker
from simulation.objects import Placement
from simulation.pick_place.bin_task import PLATFORM, BinResult, BinTask, PickRecord, replay
from simulation.pick_place.conveyor import INTERCEPT_MARGIN_S, INTERCEPT_Y, MIN_LOOKS_FOR_VELOCITY, ConveyorDemo, nominal_approach_seconds
from simulation.pick_place.demo import run_trial
from simulation.pick_place.pose_source import get_object_pose
from simulation.pick_place.scene import Scene

# Belt where both arms reach it (x = 0.38; at 0.45 only the right arm could, and
# only a can), the box between the robot and the belt, narrow (12 x 26 cm) so it
# clears the belt strip. Drops are planned to just over the rim, so a narrow box
# costs the plan nothing.
BOX = (0.23, 0.0)
BOX_HALF = (0.06, 0.13)
BOX_WALL = 0.08
BELT = Conveyor(x=0.38, width=0.08)
BELT_SPEED = -0.02          # m/s along y: from the left arm's side to the right arm's
BELT_RAMP_S = 1.0
LOOK_PERIOD_S = 0.5
BELT_PHASE_MAX_S = 150.0
# Table objects off the belt strip (x < 0.34) and clear of the box; each spot and drop
# spot plan-checked. The left arm found no grasp around (0.21 .. 0.28, 0.23 .. 0.28).
TABLE_OBJECTS = (
    Placement("can", (0.25, -0.22), "lying", 30.0),
    Placement("peach", (0.29, -0.33), name="peach"),
    Placement("apple", (0.29, 0.33), name="apple"),
    Placement("orange", (0.20, 0.38), name="orange"),
)
# On the belt, upstream: the first to arrive goes to the left arm, the second -- far
# enough behind to pass the left arm while it is busy -- to the right arm.
BELT_OBJECTS = (
    Placement("orange", (BELT.x, 0.90), name="orange_belt"),
    Placement("can", (BELT.x, 1.05), name="can_belt"),
)
BELT_PICKS = {"left": 1, "right": 1}
KNOWN = (("can", "upright"), ("can", "lying"), ("peach", "upright"), ("orange", "upright"), ("apple", "upright"))
BELT_KNOWN = (("orange", "upright"), ("can", "upright"))
DROP_SPOTS = {"right": [(0.0, -0.05), (0.0, -0.09)], "left": [(0.0, 0.09), (0.0, 0.05)]}
BELT_DROP_SPOTS = {"right": [(0.0, -0.07)], "left": [(0.0, 0.07)]}


def build_scene() -> Scene:
    objects = list(TABLE_OBJECTS) + list(BELT_OBJECTS)
    first = objects[0]
    return Scene(first.xy, BOX, work_platform_height=PLATFORM, pick_object=first.key, pick_pose=first.pose,
                 pick_yaw_deg=first.yaw_deg, extra_objects=objects[1:], basket_half_size=BOX_HALF,
                 basket_wall_height=BOX_WALL, conveyor=BELT)


class BinConveyorTask(BinTask):
    box, box_half, box_wall, known, drop_spots = BOX, BOX_HALF, BOX_WALL, KNOWN, DROP_SPOTS

    def __init__(self, *, pose_backend: str = "gt") -> None:
        super().__init__(TABLE_OBJECTS, pose_backend=pose_backend)
        belt_x = (BELT.x - 0.5 * BELT.width, BELT.x + 0.5 * BELT.width)
        # The table look stops short of the belt: its top stands 1 cm over the
        # platform, which the table detector would take for a long object.
        self.detector = ObjectDetector(self.scene.model, self.known, workspace_x=(0.17, belt_x[0] - 0.005),
                                       basket_xy=self.box, basket_margin=max(self.box_half) + 0.03)
        self.belt_detector = ObjectDetector(self.scene.model, BELT_KNOWN, workspace_x=belt_x, workspace_y=(-0.6, 0.6))
        self.belt_tracker = ObjectTracker(gate=0.06, max_missed=3)
        self.belt_started: float | None = None

    def build_scene(self) -> Scene:
        return build_scene()

    # ---------------------------------------------------------------- stepping
    def _advance(self, seconds: float) -> None:
        """Let the world run with the belt at speed (ramped over BELT_RAMP_S), recording."""
        scene = self.scene
        for _ in range(int(round(seconds / scene.model.opt.timestep))):
            ramp = min(1.0, (float(scene.data.time) - self.belt_started) / BELT_RAMP_S)
            scene.set_belt_speed(BELT_SPEED * ramp)
            mujoco.mj_step(scene.model, scene.data)
            if not self.recorder.times or float(scene.data.time) - self.recorder.times[-1] >= self.recorder.frame_seconds:
                self.recorder.sync()

    # ---------------------------------------------------------------- the belt phase
    def after_table(self, result: BinResult) -> None:
        scene = self.scene
        print("\n=== belt phase: starting the conveyor ===")

        def alone(key: str, yaw: float, pose: str = "upright") -> Scene:
            return Scene((BELT.x, 0.0), BOX, work_platform_height=PLATFORM, pick_object=key, pick_pose=pose,
                         pick_yaw_deg=yaw, basket_half_size=BOX_HALF, basket_wall_height=BOX_WALL, conveyor=BELT)

        self.belt_detector.enroll(alone)
        self.belt_started = float(scene.data.time)
        self.recorder.caption = "conveyor running"
        counts, done = {"left": 0, "right": 0}, set()
        while float(scene.data.time) - self.belt_started < BELT_PHASE_MAX_S and sum(counts.values()) < sum(BELT_PICKS.values()):
            detections = self.belt_detector.detect(scene.data, unique=False)
            tracks = self.belt_tracker.update(detections, float(scene.data.time))
            by_track = {d.track_id: d for d in detections if d.track_id is not None}
            job = self._assign(tracks, counts, done)
            if job is None:
                self._advance(LOOK_PERIOD_S)
                continue
            track, arm, intercept, start = job
            now = float(scene.data.time)
            if start - now > LOOK_PERIOD_S:
                self._advance(LOOK_PERIOD_S)
                continue
            if start > now:
                self._advance(start - now)
            detection = by_track.get(track.track_id)
            name = self.instance_of(detection) if detection is not None else self._nearest_belt_instance(track)
            print(f"[t={scene.data.time:6.1f}s] belt: #{track.track_id} {track.label} ({name}) at y={track.position[1]:+.3f} "
                  f"v={track.velocity[1] * 100:+.1f}cm/s -> {arm} arm, intercept y={INTERCEPT_Y[arm]:+.2f} at t={intercept:.1f}s")
            self.recorder.caption = f"conveyor: {arm} arm -> {track.label} ({name})"
            scene.set_target(name)
            demo = ConveyorDemo(
                scene=scene, side=arm, pose_backend=self.pose_backend, velocity=track.velocity,
                intercept_time=intercept, observe=self._observe(name), mask_source=self.mask_source_belt(name),
                place_offset=BELT_DROP_SPOTS[arm][min(counts[arm], len(BELT_DROP_SPOTS[arm]) - 1)],
            )
            trial = run_trial(demo, self.recorder)
            self._advance(1.0)
            done.add(track.track_id)
            ok = scene.object_in_basket(name)
            slip = None
            if trial.proof_lift_hand_rise_m is not None and trial.proof_lift_rise_m is not None:
                slip = round((trial.proof_lift_hand_rise_m - trial.proof_lift_rise_m) * 1000, 1)
            result.picks.append(PickRecord(
                f"{arm} (belt)", name, scene.object_types[name], 1, ok, trial.grasp_name,
                None if ok else (trial.failure_reason or "not in the box").splitlines()[0][:200],
                trial.failed_phase, slip, np.round(scene.object_position_of(name), 4).tolist(),
            ))
            print(f"   -> {'IN THE BOX' if ok else 'FAILED: ' + str(result.picks[-1].failure)}")
            self.recorder.caption = f"conveyor: {arm} arm {name} {'in the box' if ok else 'missed'}"
            if ok:
                counts[arm] += 1
                result.per_arm[arm].append(name)
            self._home(arm)

    def _assign(self, tracks, counts, done):
        now = float(self.scene.data.time)
        approach = nominal_approach_seconds() + INTERCEPT_MARGIN_S
        jobs = []
        for track in tracks:
            if track.track_id in done or track.seen < MIN_LOOKS_FOR_VELOCITY:
                continue
            vy = float(track.velocity[1])
            if vy > -1e-3:
                continue
            for arm in sorted((a for a in ("left", "right") if counts[a] < BELT_PICKS[a]), key=lambda a: (counts[a], a != "left")):
                dt = (INTERCEPT_Y[arm] - float(track.position[1])) / vy
                if dt >= approach:
                    jobs.append((now + dt - approach, track, arm, now + dt))
                    break
        if not jobs:
            return None
        start, track, arm, intercept = min(jobs, key=lambda j: j[0])
        return track, arm, intercept, start

    def _nearest_belt_instance(self, track) -> str:
        names = [p.label for p in BELT_OBJECTS if p.key == track.label]
        return min(names, key=lambda n: float(np.linalg.norm(self.scene.object_position_of(n)[:2] - track.position[:2])))

    def mask_source_belt(self, name: str):
        def fresh():
            kind = self.scene.object_types[name]
            detections = [d for d in self.belt_detector.detect(self.scene.data, unique=False) if d.label == kind]
            if not detections:
                raise RuntimeError(f"perception failed: {name} not seen on the belt")
            here = self.scene.object_position_of(name)[:2]
            return min(detections, key=lambda d: float(np.linalg.norm(d.centroid[:2] - here))).mask
        return fresh

    def _observe(self, name: str):
        source = self.mask_source_belt(name)

        def observe() -> np.ndarray:
            return get_object_pose(self.scene, name, self.pose_backend, mask=source())
        return observe


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bin task, then two more objects off a moving conveyor")
    parser.add_argument("--pose-backend", default="gt", choices=("gt", "foundationpose"))
    parser.add_argument("--frames", type=Path, default=Path("artifacts") / "bin_conveyor_frames.npz")
    parser.add_argument("--report", type=Path, default=Path("artifacts") / "bin_conveyor.json")
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.replay:
        replay(args.replay, args.speed, scene_builder=build_scene)
        return
    task = BinConveyorTask(pose_backend=args.pose_backend)
    try:
        result = task.run()
    finally:
        task.recorder.save(args.frames)  # keep the recording even if the run crashed
    from dataclasses import asdict

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(asdict(result), indent=2, default=float), encoding="utf-8")
    print("\n" + result.summary())
    print(f"recording: {args.frames} ({len(task.recorder.times)} frames); report: {args.report}")
    if not args.no_view:
        replay(args.frames, args.speed, scene_builder=build_scene)


if __name__ == "__main__":
    main()
