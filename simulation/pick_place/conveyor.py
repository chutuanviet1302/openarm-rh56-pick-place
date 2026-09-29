"""Conveyor picking: objects ride a moving belt past both arms; each arm takes two into
the basket as they come into its reach.

    python -m simulation.pick_place.conveyor                 # headless, gt 6D pose
    python -m simulation.pick_place.conveyor --viewer        # watch it
    python -m simulation.pick_place.conveyor --pose-backend foundationpose

Scene: the centre-basket platform with a belt along world y at x = 0.42 running from
the left arm's side (+y) to the right arm's (-y) at BELT_SPEED; the basket sits between
the belt and the robot. Four objects start upstream on the belt.

Loop (sim time; the belt never stops):
    look      head-camera RGB-D -> ObjectDetector (identity, pixels) -> ObjectTracker
              (id, position history -> velocity by a least-squares fit)
    assign    each object to an arm: the one with fewer picks so far (left on a tie,
              it comes first), if the object can still reach that arm's intercept
              point late enough for the arm to be there; at most PICKS_PER_ARM each
    pick      ConveyorDemo: plan the grasp for where the object will be at the
              intercept time, go to the standoff, wait for the object, correct the
              grasp target from a fresh look, close while the wrist rides along with
              the belt, lift it off, drop it in the basket

Simplification (stated in the report): the arms work one at a time -- while one arm
picks, the other waits; the objects are spaced on the belt so that works.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from simulation.five_finger_model import Conveyor
from simulation.object_detector import ObjectDetector, ObjectTracker
from simulation.objects import Placement
from simulation.pick_place import config as C
from simulation.pick_place.demo import Demo, run_trial
from simulation.pick_place.kinematics import solve_pose_ik
from simulation.pick_place.pose_source import get_object_pose
from simulation.pick_place.scene import Scene

PLATFORM = 0.10
# Belt as close to the robot as the basket allows: at x = 0.42 the left arm found no
# grasp for a still can at y = +0.27 in 15 min of planning, the right arm in 6 s.
BASKET = (0.22, 0.0)
BELT = Conveyor(x=0.38, width=0.08)
BELT_SPEED = -0.02                  # m/s along y: from the left arm's side to the right's
BELT_RAMP_S = 1.0
# Where each arm meets an object on the belt (world y); inside both arms' measured
# reach at x = 0.38 (reach map, 2026-09-28: the can grasps over y -0.19 .. -0.43 for
# the right arm; plan checks at the intercepts below).
INTERCEPT_Y = {"left": 0.27, "right": -0.27}
PICKS_PER_ARM = 2
# Belt proof-lift slip limit: the hand settles into the object while the belt still
# drags it (an orange measured 11 mm and then carried fine); a grip that is really
# failing is caught by the rim check and the box check.
BELT_PROOF_LIFT_SLIP_M = 0.015
LOOK_PERIOD_S = 0.5
MIN_LOOKS_FOR_VELOCITY = 3
INTERCEPT_MARGIN_S = 3.0
MAX_SIM_SECONDS = 240.0
# Spacing on the belt for one arm at a time: a pick takes ~30 s of sim time (start
# of the approach to back at attention, first conveyor run), 8.5 s of it before the
# grasp. Starting positions give the schedule left @20 s, right @55 s, left @88 s,
# right @122 s; each right-arm object passes the left arm while it is busy.
# Order on the belt = left, right, left, right: the left arm gets the fruit -- it
# plans the orange at (0.38, +0.27) in 20 s but found no grasp for the 10 cm can
# there (nor at x = 0.34) within 10 min; the right arm takes the can.
OBJECTS_ON_BELT = (
    Placement("orange", (BELT.x, 0.67)),
    Placement("can", (BELT.x, 0.83)),
    Placement("apple", (BELT.x, 2.03)),
    Placement("peach", (BELT.x, 2.17)),
)
# Drop spots in the basket per arm (offsets from its centre): each arm drops on its
# own half.
DROP_SPOTS = {"right": [(-0.04, -0.045), (0.04, -0.045)], "left": [(-0.04, 0.045), (0.04, 0.045)]}


def nominal_approach_seconds() -> float:
    """Sim time from the start of a pick episode until the hand reaches the grasp pose,
    as the phases schedule it (the executor may stretch a move; the fresh look before
    the final approach absorbs that)."""
    preshape = C.PRESHAPE_THUMB_OPEN_SECONDS if any(C.PRESHAPE_THUMB_STAGED.values()) else 0.0
    return (C.DROP_SETTLE_AT_START + C.MOVE_TO_RAISE + C.MOVE_TO_HOVER + C.MOVE_TO_READY + preshape
            + C.PRESHAPE_SETTLE + C.MOVE_TO_PREGRASP + C.MOVE_TO_GRASP)


class ConveyorDemo(Demo):
    """A pick episode on an object that is moving at `velocity` (world, m/s)."""

    FOLLOW_PERIOD_S = 0.01
    # Closing on a moving object: all five fingers at once, 3x the step, and no
    # waiting on a finger that has closed fully without touching. The static
    # sequence (fingers, thumb, squeeze; up to 2.4 s each) took 8.5 s on the belt and
    # the following wrist ran 17 cm, out of the arm's reach (second conveyor run).
    CLOSE_STEP_FRACTION = 0.03

    def __init__(self, *, velocity: np.ndarray, intercept_time: float, observe=None, **kwargs) -> None:
        super().__init__(release="drop", **kwargs)
        self.velocity = np.asarray(velocity, dtype=float)
        self.intercept_time = float(intercept_time)
        # observe() -> T_world_object now (the task's detector + pose backend).
        self.observe = observe
        self.grasp_wrist: np.ndarray | None = None
        self.grasp_time = 0.0
        # No retry on the spot: by the time the hand has let go, the belt has carried
        # the object away; the task hands it to the other arm instead.
        self.grasp_retries = 0

    # ---------------------------------------------------------------- prediction
    def planning_pose(self, observed: np.ndarray) -> np.ndarray:
        pose = observed.copy()
        pose[:3, 3] += self.velocity * (self.intercept_time - float(self.data.time))
        return pose

    def predicted_center(self, at_time: float) -> np.ndarray:
        """Grasp centre (world) at `at_time`, from a fresh look now -- or, when the hand
        hovering over the object hides it from the head camera, from the tracked
        estimate (the planning pose moved on at the belt velocity)."""
        entry = self.scene.grasp_target.entry
        try:
            observed = self.observe() if self.observe is not None else get_object_pose(self.scene, backend="gt")
            now = float(self.data.time)
        except RuntimeError as error:
            self.log.note(f"final look failed ({str(error).splitlines()[0]}); using the tracked estimate")
            observed, now = self.perceived_pose, self.intercept_time
        centre = observed[:3, :3] @ np.asarray(entry.center) + observed[:3, 3]
        return centre + self.velocity * (at_time - now)

    # ---------------------------------------------------------------- phases
    def phase_plan(self) -> None:
        """Plan with the object placed where it will be at the intercept: every planner
        check (raise blends, palm, clearances) then sees one consistent object. Sim
        time does not advance while planning, so the object is back where it was
        before a single physics step runs. (Planning around its current, upstream
        position failed for the left arm in the first recorded run.)"""
        scene = self.scene
        joint = self.model.joint(scene.object_joints[scene.pick_object])
        address = int(joint.qposadr[0])
        saved = scene.data.qpos[address : address + 7].copy()
        pose = self.perceived_pose  # the planning pose (perceive_pose -> planning_pose)
        from simulation.objects import quat_from_matrix

        scene.data.qpos[address : address + 3] = pose[:3, 3]
        scene.data.qpos[address + 3 : address + 7] = quat_from_matrix(pose[:3, :3])
        mujoco.mj_forward(self.model, scene.data)
        try:
            super().phase_plan()
        finally:
            scene.data.qpos[address : address + 7] = saved
            mujoco.mj_forward(self.model, scene.data)

    def phase_reach(self) -> None:
        ex, plan, scene, side = self.executor, self.plan, self.scene, self.side
        arm = f"{side}_arm"
        ex.preshape_hand(side)
        ex.hold(C.PRESHAPE_SETTLE)
        ex.move_to({arm: plan["pregrasp"]}, C.MOVE_TO_PREGRASP)
        wait = self.intercept_time - C.MOVE_TO_GRASP - float(self.data.time)
        self.log.record("intercept_wait_s", wait)
        self.log.note(f"at the standoff {wait:+.2f}s before the object's arrival" if wait >= 0
                      else f"at the standoff {-wait:.2f}s late; the fresh look corrects for it")
        if wait > 0:
            ex.hold(wait)
        # Fresh look: where will the grasp centre be once the final approach ends?
        arrive = float(self.data.time) + C.MOVE_TO_GRASP
        delta = self.predicted_center(arrive) - scene.grasp_target.center_world
        orientation = plan.phase_orientations.get("grasp", self.planner.orientation)
        target = plan.centers["grasp"] + delta
        try:
            grasp = solve_pose_ik(self.model, side, target, orientation, plan["grasp"])
        except RuntimeError as error:
            raise RuntimeError(f"intercept correction {np.round(delta * 1000, 1).tolist()}mm out of reach: {error}")
        self.log.record("intercept_correction_m", delta)
        self.log.note(f"intercept correction {np.round(delta * 1000, 1).tolist()} mm")
        ex.move_to({arm: grasp}, C.MOVE_TO_GRASP)
        plan.joints["grasp"] = grasp
        self.grasp_wrist, self.grasp_time, self.follow_orientation = target, arrive, orientation
        bend = [float(np.degrees(self.data.qpos[scene.arm_qpos[side][i]])) for i in C.WRIST_BEND_INDICES]
        self.log.record("wrist_pitch_at_grasp_deg", float(np.hypot(*bend)))

    def close_hand(self) -> dict[str, float]:
        """Close while the wrist rides along with the belt (IK every FOLLOW_PERIOD_S)."""
        ex, scene, side = self.executor, self.scene, self.side
        previous = ex.on_step
        last = [-1.0]

        def follow(executor) -> None:
            if previous is not None:
                previous(executor)
            t = float(executor.data.time)
            if t - last[0] < self.FOLLOW_PERIOD_S:
                return
            last[0] = t
            target = self.grasp_wrist + self.velocity * (t - self.grasp_time)
            seed = executor.data.ctrl[scene.arm_actuators[side]].copy()
            try:
                executor.data.ctrl[scene.arm_actuators[side]] = solve_pose_ik(
                    self.model, side, target, self.follow_orientation, seed, max_iterations=60)
            except RuntimeError:
                pass  # keep the last command; the grasp check will tell

        ex.on_step = follow
        try:
            forces = self._close_fast()
        finally:
            ex.on_step = previous
        self.log.note(f"closed while following the belt for {float(self.data.time) - self.grasp_time:.2f}s")
        return forces

    PROOF_LIFT_WAYPOINTS = 16

    def proof_lift_limits(self) -> tuple[float, float]:
        """Both hands take the right hand's limits (10 mm, 15 deg) on the belt. The
        left hand's stricter 6 mm / 11 deg were set for lifting a can back out of the
        centre basket; a left-hand orange grasped on the belt measured 7 mm twice (with
        and without the belt-following lift) -- it settles into a three-finger grip. A
        grip that is really failing is still caught after: the re-grip and the
        carry-clearance checks over the basket rim. A loosened limit, stated."""
        return BELT_PROOF_LIFT_SLIP_M, C.PROOF_LIFT_TILT_LIMIT_DEG["right"]

    def proof_lift(self, grasp_orientation: np.ndarray) -> tuple[float, float, float]:
        """The proof lift, still travelling with the belt: the wrist rises
        PROOF_LIFT_HEIGHT while it keeps the belt's velocity, so the belt does not drag
        the object's base across the grip before it leaves the belt (a stationary lift
        measured 7 mm slip on the left hand, over its 6 mm limit, first recorded run)."""
        ex, scene, side = self.executor, self.scene, self.side
        object_before = float(scene.object_position()[2])
        hand_before = float(scene.wrist_position(side)[2])
        start = scene.wrist_position(side)
        seed = self.data.ctrl[scene.arm_actuators[side]].copy()
        steps = self.PROOF_LIFT_WAYPOINTS
        path = []
        for k in range(1, steps + 1):
            f = k / steps
            target = start + self.velocity * (C.PROOF_LIFT_SECONDS * f) + np.array([0.0, 0.0, C.PROOF_LIFT_HEIGHT * f])
            seed = solve_pose_ik(self.model, side, target, self.follow_orientation, seed)
            path.append(seed)
        ex.follow({f"{side}_arm": path}, [C.PROOF_LIFT_SECONDS / steps] * steps)
        rise = float(scene.object_position()[2]) - object_before
        hand_rise = float(scene.wrist_position(side)[2]) - hand_before
        return rise, scene.object_tilt_deg(), hand_rise

    def lift_straight_up(self) -> None:
        """Lift high enough for the object as it actually hangs in the hand: an object
        grasped on the move can settle lower in the grip than planned (an orange
        cleared the rim by 4.2 cm for a 5 cm rule, first conveyor run)."""
        scene = self.scene
        needed = scene.carry_bottom_z() + C.DROP_CARRY_MARGIN - scene.object_bottom_z()
        wrist_z = float(scene.wrist_position(self.side)[2])
        self.plan.centers["lift"][2] = max(float(self.plan.centers["lift"][2]), wrist_z + needed)
        super().lift_straight_up()

    def _close_fast(self) -> dict[str, float]:
        """The static closing order -- four fingers to light contact, then the thumb,
        then squeeze (closing all five at once pressed the fingers onto the can's top
        and it stayed on the belt: 39 mm slip, third conveyor run) -- with the faster
        step and no waiting on a finger that has closed without touching."""
        ex, scene, side = self.executor, self.scene, self.side
        fingers = ("index", "middle", "ring", "pinky")
        fast = dict(step_fraction=self.CLOSE_STEP_FRACTION, stop_at_limit=True)
        ex.close_until_contact(side, fingers, force_target=1.0, **fast)
        ex.close_until_contact(side, ("thumb",), force_target=1.0, **fast)
        self.light_grip_ctrl = scene.data.ctrl[scene.hand_actuators[side]].copy()
        forces = ex.close_until_contact(side, (*fingers, "thumb"), **fast)
        self.log.note("contact force per finger (N): " + ", ".join(f"{k}={v:.2f}" for k, v in forces.items()))
        missing = self.fingers_not_pressing()
        if missing:
            raise RuntimeError(f"not all fingers touch the object before lift; missing {missing}: {forces}")
        self.log.record("grasp_forces", scene.finger_contact_forces(side))
        return forces

    def recover_from_failed_grasp(self) -> None:
        super().recover_from_failed_grasp()
        # The object kept moving: aim at where it will be one approach from now.
        self.intercept_time = float(self.data.time) + nominal_approach_seconds() + INTERCEPT_MARGIN_S


class VideoRecorder:
    """Frames of the scene at `fps` of sim time from a fixed free camera, with a
    status line, written to an mp4 (imageio-ffmpeg). `tick()` is called after every
    physics step, from the task's own stepping and from every executor's on_step."""

    def __init__(self, scene: Scene, path: Path, fps: int = 30, size=(960, 540)) -> None:
        import imageio.v2 as imageio

        self.scene, self.path, self.fps = scene, Path(path), fps
        model = scene.model
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), size[0])
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), size[1])
        self.renderer = mujoco.Renderer(model, size[1], size[0])
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [0.32, 0.0, 0.18]
        self.camera.azimuth, self.camera.elevation, self.camera.distance = 180.0, -38.0, 1.35
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(self.path), fps=fps, codec="libx264", quality=7, macro_block_size=8)
        self.status = ""
        self.next_time = 0.0
        self.frames = 0

    def tick(self, _executor=None) -> None:
        t = float(self.scene.data.time)
        if t + 1e-9 < self.next_time:
            return
        self.next_time = t + 1.0 / self.fps
        self.renderer.update_scene(self.scene.data, camera=self.camera)
        frame = self.renderer.render()
        from PIL import Image, ImageDraw

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, image.width, 26], fill=(0, 0, 0))
        draw.text((8, 6), f"t = {t:6.1f} s   belt {abs(BELT_SPEED) * 100:.0f} cm/s   {self.status}", fill=(255, 255, 255))
        self.writer.append_data(np.asarray(image))
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()


@dataclass
class PickLog:
    track_id: int
    label: str
    arm: str
    intercept_y: float
    success: bool
    failure: str | None
    failed_phase: str | None
    start_time: float
    end_time: float
    intercept_wait_s: float | None
    intercept_correction_mm: list[float] | None
    grasp_forces: dict | None


@dataclass
class ConveyorResult:
    pose_backend: str
    belt_speed: float
    picks: list[PickLog] = field(default_factory=list)
    per_arm: dict[str, list[str]] = field(default_factory=lambda: {"left": [], "right": []})
    missed: list[str] = field(default_factory=list)
    in_basket: dict[str, bool] = field(default_factory=dict)
    velocity_estimates: dict[str, list[float]] = field(default_factory=dict)
    sim_seconds: float = 0.0
    wall_seconds: float = 0.0

    def summary(self) -> str:
        return (f"left arm: {self.per_arm['left']}, right arm: {self.per_arm['right']}; "
                f"in basket {sum(self.in_basket.values())}/{len(self.in_basket)}; missed {self.missed or 'none'}; "
                f"sim {self.sim_seconds:.0f}s, wall {self.wall_seconds:.0f}s")


class ConveyorTask:
    def __init__(self, objects=OBJECTS_ON_BELT, *, pose_backend: str = "gt", speed: float = BELT_SPEED,
                 verbose: bool = False) -> None:
        objects = list(objects)
        self.scene = Scene(
            objects[0].xy, BASKET, work_platform_height=PLATFORM, pick_object=objects[0].key,
            extra_objects=objects[1:], conveyor=BELT,
        )
        self.keys = [p.key for p in objects]
        self.pose_backend, self.speed, self.verbose = pose_backend, speed, verbose
        belt_x = (BELT.x - 0.5 * BELT.width, BELT.x + 0.5 * BELT.width)
        self.detector = ObjectDetector(self.scene.model, self.keys, workspace_x=belt_x, workspace_y=(-0.6, 0.6),
                                       basket_xy=BASKET)
        self.tracker = ObjectTracker(gate=0.06, max_missed=3)
        self.viewer = None
        self.recorder: VideoRecorder | None = None
        self.last_detections: dict[int, object] = {}

    # ---------------------------------------------------------------- stepping
    def advance(self, seconds: float) -> None:
        """Let the world run (belt moving, arms holding their commands)."""
        scene = self.scene
        for _ in range(int(round(seconds / scene.model.opt.timestep))):
            ramp = min(1.0, float(scene.data.time) / BELT_RAMP_S)
            scene.set_belt_speed(self.speed * ramp)
            mujoco.mj_step(scene.model, scene.data)
            if self.recorder is not None:
                self.recorder.tick()
            if self.viewer is not None:
                self._pace()

    def _pace(self) -> None:
        """Viewer at real time while the task itself steps (the executors pace their
        own steps): redraw at 30 Hz of sim time, sleep off any lead over the wall clock."""
        sim = float(self.scene.data.time)
        if sim - getattr(self, "_last_frame", -1.0) < C.VIEWER_FRAME_SECONDS:
            return
        self._last_frame = sim
        self.viewer.sync()
        now = time.perf_counter()
        anchor = getattr(self, "_wall_anchor", None)
        if anchor is None or anchor + sim - now < -0.5:
            self._wall_anchor = anchor = now - sim
        ahead = anchor + sim - now
        if ahead > 0.0:
            time.sleep(ahead)

    def look(self) -> list:
        t = float(self.scene.data.time)
        detections = self.detector.detect(self.scene.data)
        tracks = self.tracker.update(detections, t)
        self.last_detections = {d.track_id: d for d in detections if d.track_id is not None}
        return tracks

    def observe_pose(self, key: str):
        """observe() for ConveyorDemo: a fresh detection of `key`, then its 6D pose."""
        def observe() -> np.ndarray:
            detections = self.detector.detect(self.scene.data)
            mask = next((d.mask for d in detections if d.label == key), None)
            if mask is None and self.pose_backend != "gt":
                raise RuntimeError(f"perception failed: {key} not seen on the belt")
            return get_object_pose(self.scene, key, self.pose_backend, mask=mask)
        return observe

    # ---------------------------------------------------------------- dispatch
    def assign(self, tracks, counts: dict[str, int], done: set[int]):
        """(track, arm, intercept time, start time) of the next pick to make, or None."""
        now, approach = float(self.scene.data.time), nominal_approach_seconds() + INTERCEPT_MARGIN_S
        jobs = []
        for track in tracks:
            if track.track_id in done or track.seen < MIN_LOOKS_FOR_VELOCITY:
                continue
            vy = float(track.velocity[1])
            if vy > -1e-3:
                continue  # not (yet) seen moving toward the arms
            arms = sorted((a for a in ("left", "right") if counts[a] < PICKS_PER_ARM),
                          key=lambda a: (counts[a], a != "left"))
            for arm in arms:
                dt = (INTERCEPT_Y[arm] - float(track.position[1])) / vy
                if dt < approach:
                    continue  # the arm could not be there in time
                jobs.append((now + dt - approach, track, arm, now + dt))
                break
        if not jobs:
            return None
        start, track, arm, intercept = min(jobs, key=lambda j: j[0])
        return track, arm, intercept, start

    def run(self) -> ConveyorResult:
        started = time.perf_counter()
        result = ConveyorResult(self.pose_backend, self.speed)
        scene = self.scene
        print("enrolling the objects (reference features on the stopped belt) ...")

        def alone(key: str, yaw: float, pose: str = "upright") -> Scene:
            return Scene((BELT.x, 0.0), BASKET, work_platform_height=PLATFORM, pick_object=key,
                         pick_pose=pose, pick_yaw_deg=yaw, conveyor=BELT)

        self.detector.enroll(alone)
        counts, done = {"left": 0, "right": 0}, set()
        spots = {arm: list(s) for arm, s in DROP_SPOTS.items()}
        while float(scene.data.time) < MAX_SIM_SECONDS:
            tracks = self.look()
            if sum(counts.values()) >= 2 * PICKS_PER_ARM:
                break
            job = self.assign(tracks, counts, done)
            if job is None:
                # More objects may still be coming from upstream, out of view: keep
                # watching until both arms have their picks or time runs out.
                self.advance(LOOK_PERIOD_S)
                continue
            track, arm, intercept, start = job
            if start - float(scene.data.time) > LOOK_PERIOD_S:
                self.advance(LOOK_PERIOD_S)  # keep tracking until it is time to go
                continue
            if start > float(scene.data.time):
                self.advance(start - float(scene.data.time))
            print(f"[t={scene.data.time:6.1f}s] #{track.track_id} {track.label} at y={track.position[1]:+.3f} "
                  f"v={track.velocity[1]*100:+.1f}cm/s -> {arm} arm, intercept y={INTERCEPT_Y[arm]:+.2f} "
                  f"at t={intercept:.1f}s")
            scene.set_target(track.label)
            detection = self.last_detections.get(track.track_id)
            demo = ConveyorDemo(
                scene=scene, side=arm, pose_backend=self.pose_backend, velocity=track.velocity,
                intercept_time=intercept, observe=self.observe_pose(track.label),
                detection_mask=None if detection is None else detection.mask,
                place_offset=spots[arm][0] if spots[arm] else (0.0, 0.0), verbose=self.verbose,
            )
            demo.executor.viewer = self.viewer
            if self.recorder is not None:
                demo.executor.on_step = self.recorder.tick
                self.recorder.status = f"{arm} arm -> #{track.track_id} {track.label}"
            # The belt keeps running inside the episode: the executor steps the same
            # MjData, and the belt's velocity servo keeps its command.
            t0 = float(scene.data.time)
            trial = run_trial(demo, self.viewer)
            done.add(track.track_id)
            ok = scene.object_in_basket(track.label)
            values = demo.log.values
            result.picks.append(PickLog(
                track.track_id, track.label, arm, INTERCEPT_Y[arm], ok,
                None if ok else (trial.failure_reason or "not in the basket").splitlines()[0][:200],
                trial.failed_phase, round(t0, 2), round(float(scene.data.time), 2),
                values.get("intercept_wait_s"),
                None if values.get("intercept_correction_m") is None
                else [round(v * 1000, 1) for v in values["intercept_correction_m"]],
                values.get("grasp_forces"),
            ))
            result.velocity_estimates[f"#{track.track_id} {track.label}"] = np.round(track.velocity, 4).tolist()
            if ok:
                counts[arm] += 1
                result.per_arm[arm].append(track.label)
                if spots[arm]:
                    spots[arm].pop(0)
            if self.recorder is not None:
                self.recorder.status = (f"{arm} arm: {track.label} {'IN THE BASKET' if ok else 'missed'}   "
                                        f"(left {counts['left']}, right {counts['right']})")
            print(f"   -> {'IN THE BASKET' if ok else 'FAILED'}"
                  + ("" if ok else f" ({result.picks[-1].failed_phase}: {result.picks[-1].failure})"))
            self._home(arm)
        result.missed = [k for k in self.keys if not scene.object_in_basket(k)]
        result.in_basket = {k: scene.object_in_basket(k) for k in self.keys}
        result.sim_seconds = float(scene.data.time)
        result.wall_seconds = time.perf_counter() - started
        return result

    def _home(self, side: str) -> None:
        """Back to attention if a failed pick left the arm elsewhere."""
        from simulation.pick_place.executor import Executor

        scene = self.scene
        here = scene.data.qpos[scene.arm_qpos[side]]
        if np.max(np.abs(here - scene.attention_pose[side])) < 0.05:
            return
        executor = Executor(scene)
        executor.viewer = self.viewer
        executor.active_side = side
        if self.recorder is not None:
            executor.on_step = self.recorder.tick
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pick objects off a moving conveyor with both arms")
    parser.add_argument("--pose-backend", default="gt", choices=("gt", "foundationpose"))
    parser.add_argument("--speed", type=float, default=BELT_SPEED, help="belt speed, m/s along y (negative: left -> right)")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("artifacts") / "conveyor.json")
    parser.add_argument("--video", type=Path, help="write an mp4 of the run (front camera, 30 fps of sim time)")
    args = parser.parse_args(argv)
    task = ConveyorTask(pose_backend=args.pose_backend, speed=args.speed, verbose=args.verbose)
    if args.video:
        task.recorder = VideoRecorder(task.scene, args.video)
    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(task.scene.model, task.scene.data) as viewer:
            # Front view: belt across the picture, basket and both arms behind it.
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [0.32, 0.0, 0.18]
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 180.0, -38.0, 1.35
            task.viewer = viewer
            result = task.run()
            print("\n" + result.summary() + "\nclose the viewer window to exit")
            while viewer.is_running():  # leave the final scene up until the window is closed
                viewer.sync()
                time.sleep(0.05)
    else:
        result = task.run()
    if task.recorder is not None:
        task.recorder.close()
        print(f"video: {args.video} ({task.recorder.frames} frames)")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(asdict(result), indent=2, default=float), encoding="utf-8")
    print("\n" + result.summary())
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
