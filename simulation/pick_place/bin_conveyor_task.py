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

from simulation.five_finger_model import BASKET_WALL_THICKNESS, WORK_PLATFORM_X, Conveyor
from simulation.object_detector import ObjectDetector, ObjectTracker
from simulation.objects import Placement, footprint_radius
from simulation.pick_place import config as C
from simulation.pick_place.bin_task import PLATFORM, BinResult, BinTask, PickRecord, add_replay_arguments, replay
from simulation.pick_place.conveyor import INTERCEPT_MARGIN_S, INTERCEPT_Y, MIN_LOOKS_FOR_VELOCITY, ConveyorDemo, nominal_approach_seconds
from simulation.pick_place.demo import run_trial
from simulation.pick_place.layout import HAND_HALF_WIDTH, box_clearance_ok, derive_box
from simulation.pick_place.perception_loop import Perception
from simulation.pick_place.pose_source import get_object_pose
from simulation.pick_place.scene import Scene

# Belt where both arms reach it (x = 0.38; at 0.45 only the right arm could, and
# only a can). The box stands between the robot and the belt, derived from the
# fixtures (layout.derive_box): on the work platform -- the old box (centre x 0.23)
# overhung its edge by 3 cm into the resting left thumb (0.77 mm contact in 8740 of
# 11724 frames, scripts/task_metrics, 2026-10-01) -- and PATH_CLEARANCE short of the
# belt strip. Drops are planned to just over the rim.
BELT = Conveyor(x=0.38, width=0.08)
BOX, BOX_HALF = derive_box(WORK_PLATFORM_X, BELT)
BOX_WALL = 0.08
BELT_SPEED = -0.02          # m/s along y: from the left arm's side to the right arm's
BELT_RAMP_S = 1.0
LOOK_PERIOD_S = 0.5
BELT_PHASE_MAX_S = 150.0
# Table objects off the belt strip (x < 0.34) and clear of the box; each spot and drop
# spot plan-checked. The left arm found no grasp around (0.21 .. 0.28, 0.23 .. 0.28).
# Every object keeps half its arm's hand width from the box wall, the rule the random
# layouts follow (layout.table_region): the lying can at y = -0.28 (9 mm from the
# wall), then -0.292 (PATH_CLEARANCE), and the right index finger, wrapping it, met
# the wall both times (0.66 / 0.56 mm, 2026-10-01). So the can sits at that
# clearance and the peach keeps its 0.11 m from it.
_LYING_CAN = Placement("can", (0.0, 0.0), "lying", 30.0)
_CAN_Y = -float(np.ceil((BOX_HALF[1] + BASKET_WALL_THICKNESS + HAND_HALF_WIDTH["right"]
                         + footprint_radius(_LYING_CAN)) * 1000) / 1000)  # mm, away from the box
TABLE_OBJECTS = (
    Placement("can", (0.25, _CAN_Y), "lying", 30.0),
    Placement("peach", (0.24, round(_CAN_Y - 0.11, 3)), name="peach"),
    Placement("apple", (0.29, 0.33), name="apple"),
    Placement("orange", (0.20, 0.38), name="orange"),
)
for _p in TABLE_OBJECTS:  # the clearance rule holds for every table object (its arm: its side)
    assert box_clearance_ok(_p, BOX_HALF, "right" if _p.xy[1] < 0 else "left"), _p
# On the belt, two cans, both for the right arm: a belt orange was lost in every
# run (left hand: three-finger grip; right hand: 11 mm proof-lift slip), a belt can
# never. Both start upstream, out of the belt camera's view (y > BELT_VIEW_Y + the
# can's radius): one waiting at y = 0.30 stood next to the apple and the left index
# finger touched it picking the apple (4.7 mm, run r3, 2026-10-01). The second is as
# far behind the first as before (0.65 m: one pick and the way home between them).
BELT_VIEW_Y = 0.6
_BELT_START = BELT_VIEW_Y + footprint_radius("can") + 0.01
BELT_OBJECTS = (
    Placement("can", (BELT.x, round(_BELT_START, 3)), name="can_belt_2"),
    Placement("can", (BELT.x, round(_BELT_START + 0.65, 3)), name="can_belt"),
)
# Belt picks per arm (at most) and in all: the arm with fewer picks takes the next
# object (left on a tie); an object one arm missed goes on to the other one (if allowed).
BELT_PICKS = {"left": 0, "right": 2}
BELT_FIRST_TRACK_ID = 100   # belt track ids start here (the table tracker counts from 1)
BELT_TOTAL = 2
KNOWN = (("can", "upright"), ("can", "lying"), ("peach", "upright"), ("orange", "upright"), ("apple", "upright"))
BELT_KNOWN = (("orange", "upright"), ("can", "upright"))
# Drop spots: chosen by the planner among layout.drop_spot_candidates (none fixed).
DROP_SPOTS = None
BELT_DROP_SPOTS = None


def build_scene() -> Scene:
    objects = list(TABLE_OBJECTS) + list(BELT_OBJECTS)
    first = objects[0]
    return Scene(first.xy, BOX, work_platform_height=PLATFORM, pick_object=first.key, pick_pose=first.pose,
                 pick_yaw_deg=first.yaw_deg, extra_objects=objects[1:], basket_half_size=BOX_HALF,
                 basket_wall_height=BOX_WALL, conveyor=BELT)


def scene_from_layout(raw: dict) -> Scene:
    """The scene of a recorded run (its layout JSON, saved with the frames)."""
    from simulation.pick_place.layout import Layout

    return Layout.from_json(json.dumps(raw)).scene()


class BinConveyorTask(BinTask):
    box, box_half, box_wall, known, drop_spots = BOX, BOX_HALF, BOX_WALL, KNOWN, DROP_SPOTS
    keep_over_last = False      # the left arm is done after the table: it goes home at once

    def __init__(self, *, pose_backend: str = "foundationpose", motion: str = "waypoints", layout=None) -> None:
        """`layout`: a layout.Layout (random, seed-drawn) -- box, objects, and no fixed
        drop spots or belt-arm split; None keeps the fixed demo layout of this module."""
        self.spec = layout
        if layout is not None:
            self.box, self.box_half = tuple(layout.box_xy), tuple(layout.box_half)
            self.drop_spots = None
            self.picks_per_arm = len(layout.table)
            self.belt_objects = list(layout.belt)
            self.belt_known = tuple(sorted({(p.key, "upright") for p in layout.belt}))
            self.belt_picks = {"left": len(layout.belt), "right": len(layout.belt)}
            self.belt_drop_spots = None
        else:
            self.belt_objects, self.belt_known = list(BELT_OBJECTS), BELT_KNOWN
            self.belt_picks, self.belt_drop_spots = dict(BELT_PICKS), BELT_DROP_SPOTS
        super().__init__(layout.table if layout is not None else TABLE_OBJECTS, pose_backend=pose_backend, motion=motion)
        belt_x = (BELT.x - 0.5 * BELT.width, BELT.x + 0.5 * BELT.width)
        # The table look stops short of the belt: its top stands 1 cm over the
        # platform, which the table detector would take for a long object.
        self.detector = ObjectDetector(self.scene.model, self.known, workspace_x=(0.17, belt_x[0] - 0.005),
                                       basket_xy=self.box, basket_margin=max(self.box_half) + 0.03)
        self.belt_detector = ObjectDetector(self.scene.model, self.belt_known, workspace_x=belt_x,
                                            workspace_y=(-BELT_VIEW_Y, BELT_VIEW_Y))
        self.belt_tracker = ObjectTracker(gate=0.06, max_missed=3, first_id=BELT_FIRST_TRACK_ID)
        self.belt_started: float | None = None
        self.perception = self.build_perception()
        self.recorder.hooks[:] = [self.perception.on_frame]

    def build_perception(self) -> Perception:
        if not hasattr(self, "belt_detector"):  # BinTask.__init__ runs first; replaced above
            return super().build_perception()
        return Perception(self.scene, self.detector, self.tracker, self.belt_detector, self.belt_tracker,
                          period=LOOK_PERIOD_S)

    def enroll(self) -> None:
        super().enroll()

        def alone(key: str, yaw: float, pose: str = "upright") -> Scene:
            # Mid-way along the belt's visible stretch (at y = 0 the object stood within
            # the 15 cm the scene builder keeps between an object and the box).
            return Scene((BELT.x, 0.5 * BELT_VIEW_Y), self.box, work_platform_height=PLATFORM, pick_object=key, pick_pose=pose,
                         pick_yaw_deg=yaw, basket_half_size=self.box_half, basket_wall_height=BOX_WALL, conveyor=BELT)

        self.belt_detector.enroll(alone)

    def build_scene(self) -> Scene:
        return build_scene() if self.spec is None else self.spec.scene()

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
        self.belt_started = float(scene.data.time)
        self.recorder.caption = "conveyor running"
        counts, done = {"left": 0, "right": 0}, set()
        tried: dict[int, set] = {}
        while (float(scene.data.time) - self.belt_started < BELT_PHASE_MAX_S
               and sum(counts.values()) < len(self.belt_objects) and not self.belt_done()):
            # The look the recorder hook took during the last advance (one period old
            # at most), or a fresh one.
            detections = self.perception.look(max_age=LOOK_PERIOD_S).detections.get("belt", [])
            tracks = [t for t in self.belt_tracker.tracks.values() if t.missed == 0]
            by_track = {d.track_id: d for d in detections if d.track_id is not None}
            job = self._assign(tracks, counts, done, tried)
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
            # One arm at a time: the other must be at attention; this one may start
            # straight from over the box.
            other = "right" if arm == "left" else "left"
            if other in self.arm_over:
                self.arm_over.pop(other).return_home()
                continue  # that took a while: look and plan again
            scene.set_target(name)
            if self.belt_drop_spots is not None:
                spot, candidates = self.belt_drop_spots[arm][min(counts[arm], len(self.belt_drop_spots[arm]) - 1)], None
            else:
                spot, candidates = None, self.drop_candidates(arm, name)
            demo = ConveyorDemo(
                scene=scene, side=arm, pose_backend=self.pose_backend, velocity=track.velocity,
                intercept_time=intercept, observe=self._observe(name), mask_source=self.mask_source_belt(name),
                place_offset=spot, place_candidates=candidates,
                # Home after a belt pick: an arm parked over the box hides the belt
                # from the head camera and the next object is never tracked.
                stay_over_basket=False, start_over_basket=arm in self.arm_over, motion=self.motion,
            )
            self.arm_over.pop(arm, None)
            trial = run_trial(demo, self.recorder)
            self._advance(1.0)
            tried.setdefault(track.track_id, set()).add(arm)
            ok = scene.object_in_basket(name)
            if ok:
                done.add(track.track_id)
            slip = None
            if trial.proof_lift_hand_rise_m is not None and trial.proof_lift_rise_m is not None:
                slip = round((trial.proof_lift_hand_rise_m - trial.proof_lift_rise_m) * 1000, 1)
            result.picks.append(PickRecord(
                f"{arm} (belt)", name, scene.object_types[name], 1, ok, trial.grasp_name,
                None if ok else (trial.failure_reason or "not in the box").splitlines()[0][:200],
                trial.failed_phase, slip, np.round(scene.object_position_of(name), 4).tolist(),
            ))
            print(f"   -> {'IN THE BOX' if ok else 'FAILED: ' + str(result.picks[-1].failure)}")
            if trial.failure_reason:
                print(f"   (episode error: {trial.failure_reason.splitlines()[0][:200]})")
            self.recorder.caption = f"conveyor: {arm} arm {name} {'in the box' if ok else 'missed'}"
            self.on_pick(name, ok)
            if ok:
                counts[arm] += 1
                result.per_arm[arm].append(name)
                if demo.place_offset is not None:
                    self.dropped[arm].append(tuple(float(v) for v in demo.place_offset[:2]))
            self._home(arm)
        for demo in self.arm_over.values():
            demo.return_home()
        self.arm_over.clear()

    def _assign(self, tracks, counts, done, tried=None):
        now = float(self.scene.data.time)
        approach = nominal_approach_seconds() + INTERCEPT_MARGIN_S
        jobs = []
        for track in tracks:
            if track.track_id in done or track.seen < MIN_LOOKS_FOR_VELOCITY or not self.wanted(track.label):
                continue
            vy = float(track.velocity[1])
            if vy > -1e-3:
                continue
            arms = [a for a in ("left", "right") if counts[a] < self.belt_picks[a]
                    and a not in (tried or {}).get(track.track_id, ()) and self.belt_arm_can(a, track.label)]
            for arm in sorted(arms, key=lambda a: (counts[a], a != "left")):
                dt = (INTERCEPT_Y[arm] - float(track.position[1])) / vy
                if dt >= approach:
                    jobs.append((now + dt - approach, track, arm, now + dt))
                    break
        if not jobs:
            return None
        start, track, arm, intercept = min(jobs, key=lambda j: j[0])
        return track, arm, intercept, start

    def belt_done(self) -> bool:
        """End the belt phase early (product.cell: the order is complete)."""
        return False

    def belt_arm_can(self, arm: str, kind: str) -> bool:
        """Can this arm plan a `kind` standing on the belt at its intercept point? Checked
        once per (arm, kind) by the real planner in a time-bounded process (fixed
        layout: BELT_PICKS decides, as measured before)."""
        if self.spec is None:
            return True
        cache = self.__dict__.setdefault("_belt_reach", {})
        if (arm, kind) not in cache:
            from simulation.pick_place.layout import Layout, plan_check_isolated

            probe = Placement(kind, (BELT.x, INTERCEPT_Y[arm]), "upright")  # first object: named by its key
            layout = Layout(None, self.box, self.box_half, [probe], [])
            ok, detail = plan_check_isolated(layout, kind, arm)
            cache[(arm, kind)] = ok
            print(f"   belt reach: {arm} arm {'plans' if ok else 'cannot plan'} a {kind} at y={INTERCEPT_Y[arm]:+.2f} ({detail})")
        return cache[(arm, kind)]

    def _nearest_belt_instance(self, track) -> str:
        names = [p.label for p in self.belt_objects if p.key == track.label]
        return min(names, key=lambda n: float(np.linalg.norm(self.scene.object_position_of(n)[:2] - track.position[:2])))

    def mask_source_belt(self, name: str):
        def fresh():
            kind = self.scene.object_types[name]
            detections = [d for d in self.perception.look().detections.get("belt", []) if d.label == kind]
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


def _builder_for(frames: Path):
    """The scene builder for a recording: its saved layout, else the fixed layout."""
    recording = np.load(frames)
    if "layout" in recording.files:
        raw = json.loads(str(recording["layout"]))
        return lambda: scene_from_layout(raw)
    return build_scene


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bin task, then two more objects off a moving conveyor")
    parser.add_argument("--pose-backend", default="foundationpose", choices=("gt", "foundationpose"))
    parser.add_argument("--frames", type=Path, default=Path("artifacts") / "bin_conveyor_frames.npz")
    parser.add_argument("--report", type=Path, default=Path("artifacts") / "bin_conveyor.json")
    parser.add_argument("--motion", default="waypoints", choices=("waypoints", "mink"))
    parser.add_argument("--seed", type=int, help="draw a random layout (layout.sample_layout); default: the fixed layout")
    parser.add_argument("--layout-file", type=Path, help="a layout JSON saved by an earlier --seed run")
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--replay", type=Path)
    add_replay_arguments(parser)
    args = parser.parse_args(argv)
    if args.replay:
        replay(args.replay, args.speed, scene_builder=_builder_for(args.replay), quality=args.quality, show_ui=args.ui,
               hold=not args.no_hold)
        return
    layout = None
    if args.layout_file:
        from simulation.pick_place.layout import Layout

        layout = Layout.from_json(args.layout_file.read_text(encoding="utf-8"))
    elif args.seed is not None:
        from simulation.pick_place.layout import sample_layout

        print(f"drawing layout for seed {args.seed} ...")
        layout = sample_layout(args.seed)
        out = args.frames.with_name(args.frames.name.replace("_frames.npz", "_layout.json"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(layout.to_json(), encoding="utf-8")
        print(f"layout: {out}")
    task = BinConveyorTask(pose_backend=args.pose_backend, motion=args.motion, layout=layout)
    try:
        result = task.run()
    finally:
        if layout is not None:
            task.recorder.extra["layout"] = np.asarray(layout.to_json())
        task.recorder.save(args.frames)  # keep the recording even if the run crashed
    from dataclasses import asdict

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(asdict(result), indent=2, default=float), encoding="utf-8")
    print("\n" + result.summary())
    print(f"recording: {args.frames} ({len(task.recorder.times)} frames); report: {args.report}")
    if not args.no_view:
        replay(args.frames, args.speed, scene_builder=_builder_for(args.frames), quality=args.quality, show_ui=args.ui,
               hold=not args.no_hold)


if __name__ == "__main__":
    main()
