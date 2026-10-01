"""Continuous perception for a task: the head camera looks every `period` s of sim time,
the table and the belt detectors share that one RGB-D frame, each feeds its tracker,
every identified object gets a 6D pose from the pose backend (FoundationPose), and
every look is logged so the replay can show what the robot saw.

    perception = Perception(scene, table_detector, table_tracker, belt_detector, belt_tracker,
                            phase_source=lambda: recorder.phase, backend="foundationpose",
                            instance_of=task.instance_of)
    recorder.hooks.append(perception.on_frame)     # look while the task runs
    look = perception.look()                        # a fresh look for a decision
    perception.estimate_6d("table", look.detections["table"])   # 6D for the new objects
    perception.set_target("right", detection)       # the object a pick is about to take
    recorder.extra.update(perception.log.arrays())  # saved with the replay frames

What is shown (and logged):
    6D pose              each identified object once the pose backend has estimated it
                         (all table objects at the first look, in one FoundationPose
                         call; a belt object once its track is steady; again at every
                         pick): its model's box at that pose plus x/y/z axes. A table
                         object keeps its estimate; a belt object's estimate moves on
                         with its track (FoundationPose once, the tracker after).
    tracking only        an identified object not estimated yet: the box of its depth
                         points (world axis-aligned), no axes
    the held object      from the grasp until the drop (demo phases "grasp", "carry"):
                         the object's 6D pose (or tracked box) at the grasp, fixed to the
                         hand (hand pose from the arm's joint angles); detections
                         overlapping it are left out of the display (partial views
                         between the fingers); the trackers still get every detection.
    not shown            candidates that match no known object; the robot's own arm and
                         hand pixels are removed before looking (object_detector)

Replay: DetectionReplayOverlay draws the latest look's objects (6D boxes + axes, or
tracked boxes), the held object recomputed from the hand in every frame, and the
annotated camera image as an inset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from simulation.fp_bridge import HEIGHT, WIDTH
from simulation.object_detector import CameraFrame, Detection, ObjectDetector, ObjectTracker
from simulation.pick_place.pose_overlay import BOX_EDGES, INSET_SIZE, object_box

REGIONS = ("table", "belt")
HELD_PHASES = ("grasp", "carry")
SIDES = ("left", "right")
# Image colours (RGB) and marker colours (RGBA) per region.
REGION_RGB = {"table": (40, 200, 255), "belt": (255, 170, 0), "held": (60, 230, 90)}
REGION_RGBA = {"table": (0.15, 0.8, 1.0, 1.0), "belt": (1.0, 0.65, 0.0, 1.0), "held": (0.25, 0.95, 0.35, 1.0)}
AXIS_RGB = ((255, 40, 40), (40, 220, 40), (40, 90, 255))           # x, y, z
AXIS_RGBA = ((1.0, 0.1, 0.1, 1.0), (0.1, 0.9, 0.1, 1.0), (0.1, 0.3, 1.0, 1.0))
AXIS_LENGTH_M = 0.06
# The 3-D viewer looks from ~1.5 m: longer, thicker axes and box lines there.
REPLAY_AXIS_LENGTH_M = 0.12
REPLAY_AXIS_WIDTH_M = 0.005
REPLAY_LINE_WIDTH_PX = 4.0
# Box extent from the detection's depth points: these percentiles, not min/max, so the
# few mixed-depth pixels on a silhouette edge do not stretch the box.
EXTENT_PERCENTILES = (1.0, 99.0)
VELOCITY_ARROW_S = 2.0      # the belt arrow shows where the object will be this much later
_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)


def _box_frame(pose: np.ndarray, local_box: np.ndarray) -> np.ndarray:
    """4x4 of the model box's centre and axes in the world, for an object at `pose`."""
    frame = np.eye(4)
    frame[:3, :3] = pose[:3, :3]
    frame[:3, 3] = pose[:3, :3] @ local_box[:3] + pose[:3, 3]
    return frame


def _corners(frame: np.ndarray, half: np.ndarray) -> np.ndarray:
    return frame[:3, 3] + (_SIGNS * half) @ frame[:3, :3].T


def _aabb(corners: np.ndarray) -> np.ndarray:
    return np.stack([corners.min(axis=0), corners.max(axis=0)])


@dataclass
class Held:
    """The object in a hand: its box centre and axes in the hand (EE site) frame and its
    half size; `six_d`: from a 6D estimate (else the tracked box, no orientation)."""
    side: str
    label: str
    track: int
    offset: np.ndarray        # box centre in the hand frame
    rotation: np.ndarray      # box axes in the hand frame (3x3)
    half: np.ndarray          # box half-size along its axes
    six_d: bool = False

    def frame(self, hand_position: np.ndarray, hand_rotation: np.ndarray) -> np.ndarray:
        frame = np.eye(4)
        frame[:3, :3] = hand_rotation @ self.rotation
        frame[:3, 3] = hand_position + hand_rotation @ self.offset
        return frame

    def box(self, hand_position: np.ndarray, hand_rotation: np.ndarray) -> np.ndarray:
        """World axis-aligned (2, 3) min / max of the held box for this hand pose."""
        return _aabb(_corners(self.frame(hand_position, hand_rotation), self.half))


@dataclass
class Shown:
    """One object as displayed: its axis-aligned box always; with a 6D estimate, also
    the model box's frame (centre + axes) and half size."""
    region: str
    label: str
    track: int
    cost: float
    box: np.ndarray                      # (2, 3) world min / max
    centroid: np.ndarray
    velocity: np.ndarray
    rect: tuple[int, int, int, int] | None = None   # image rectangle (tracking-only display)
    frame: np.ndarray | None = None       # 4x4 model box frame (6D)
    half: np.ndarray | None = None
    anchor: np.ndarray | None = None      # held: side, offset(3), rotation(9), half(3), six_d


@dataclass
class Look:
    t: float
    frame: CameraFrame
    detections: dict[str, list[Detection]] = field(default_factory=dict)
    held: list[tuple[Held, np.ndarray]] = field(default_factory=list)   # (held, world box now)
    # Objects partly hidden by the robot, shown under the identity seen at that place in
    # the last look that saw them whole: region -> [(detection, label, track)].
    occluded: dict[str, list[tuple[Detection, str, int]]] = field(default_factory=dict)


def detection_box(frame: CameraFrame, detection: Detection) -> np.ndarray:
    points = frame.points[detection.mask]
    points = points[np.isfinite(points).all(axis=1)]
    return np.percentile(points, EXTENT_PERCENTILES, axis=0)


def _overlaps_xy(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a[0, :2] <= b[1, :2]) and np.all(b[0, :2] <= a[1, :2]))


def _project(points: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    world_to_cam = np.linalg.inv(T_world_cam)
    cam = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    uv = cam @ K.T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)


class DetectionLog:
    """Every look's shown objects, flattened for np.savez (one row per object)."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.rows: list[Shown] = []
        self.look_index: list[int] = []
        self.images: list[np.ndarray] = []

    def add(self, look: Look, shown: list[Shown], camera_K, T_world_cam) -> None:
        import cv2

        index = len(self.times)
        self.times.append(look.t)
        image = cv2.cvtColor(look.frame.rgb, cv2.COLOR_RGB2BGR)
        for item in shown:
            self.rows.append(item)
            self.look_index.append(index)
            colour = REGION_RGB[item.region][::-1]
            if item.frame is not None:
                uv = _project(_corners(item.frame, item.half), camera_K, T_world_cam).astype(int)
                for a, b in BOX_EDGES:
                    cv2.line(image, tuple(uv[a]), tuple(uv[b]), colour, 2, cv2.LINE_AA)
                ends = item.frame[:3, 3] + (item.frame[:3, :3] * AXIS_LENGTH_M).T
                axes = _project(np.vstack([item.frame[:3, 3], ends]), camera_K, T_world_cam).astype(int)
                for k in range(3):
                    cv2.arrowedLine(image, tuple(axes[0]), tuple(axes[k + 1]), AXIS_RGB[k][::-1], 2, cv2.LINE_AA,
                                    tipLength=0.2)
                anchor_xy = (int(uv[:, 0].min()), int(uv[:, 1].min()))
            else:
                rect = item.rect
                if rect is None:  # a held box without 6D: its projected corners
                    uv = _project(item.box[0] + (_SIGNS * 0.5 + 0.5) * (item.box[1] - item.box[0]), camera_K, T_world_cam)
                    rect = (int(uv[:, 0].min()), int(uv[:, 1].min()), int(uv[:, 0].max()), int(uv[:, 1].max()))
                cv2.rectangle(image, rect[:2], rect[2:], colour, 2)
                anchor_xy = rect[:2]
            text = (f"#{item.track} " if item.track >= 0 else "") + item.label + (" 6D" if item.frame is not None else "")
            for thick, ink in ((3, (0, 0, 0)), (1, colour)):
                cv2.putText(image, text, (anchor_xy[0], max(12, anchor_xy[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ink,
                            thick, cv2.LINE_AA)
            if np.linalg.norm(item.velocity) > 0.0:
                ends = np.vstack([item.centroid, item.centroid + item.velocity * VELOCITY_ARROW_S])
                uv = _project(ends, camera_K, T_world_cam).astype(int)
                cv2.arrowedLine(image, tuple(uv[0]), tuple(uv[1]), colour, 2, cv2.LINE_AA, tipLength=0.25)
        counts = {r: sum(1 for s in shown if s.region == r) for r in (*REGIONS, "held")}
        six_d = sum(1 for s in shown if s.frame is not None)
        caption = (f"t={look.t:5.1f}s  table {counts['table']}  belt {counts['belt']}  held {counts['held']}"
                   f"  6D {six_d}")
        for thick, ink in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(image, caption, (8, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, thick, cv2.LINE_AA)
        image = cv2.resize(image, INSET_SIZE, interpolation=cv2.INTER_AREA)
        self.images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self.rows)
        nan16, nan3 = np.full(16, np.nan), np.full(3, np.nan)
        return dict(
            det_look_times=np.asarray(self.times, dtype=float),
            det_look=np.asarray(self.look_index, dtype=int),
            det_region=np.asarray([r.region for r in self.rows], dtype=str),
            det_label=np.asarray([r.label for r in self.rows], dtype=str),
            det_track=np.asarray([r.track for r in self.rows], dtype=int),
            det_cost=np.asarray([r.cost for r in self.rows], dtype=float),
            det_box=np.asarray([r.box for r in self.rows], dtype=float).reshape(n, 2, 3),
            det_centroid=np.asarray([r.centroid for r in self.rows], dtype=float).reshape(n, 3),
            det_velocity=np.asarray([r.velocity for r in self.rows], dtype=float).reshape(n, 3),
            det_frame=np.asarray([r.frame.ravel() if r.frame is not None else nan16 for r in self.rows]).reshape(n, 16),
            det_half=np.asarray([r.half if r.half is not None else nan3 for r in self.rows], dtype=float).reshape(n, 3),
            det_anchor=np.asarray([r.anchor if r.anchor is not None else np.full(17, np.nan) for r in self.rows],
                                  dtype=float).reshape(n, 17),
            det_images=np.asarray(self.images, dtype=np.uint8).reshape(len(self.images), INSET_SIZE[1], INSET_SIZE[0], 3),
        )


def hand_pose(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> tuple[np.ndarray, np.ndarray]:
    """The hand (EE site) pose from the arm's current kinematics."""
    from simulation.pick_place.config import EE_SITE

    site = model.site(EE_SITE[side]).id
    return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()


class Perception:
    """One camera, up to two detectors (table, belt) with their trackers, and the 6D
    poses the pose backend gave for the tracked objects."""

    def __init__(self, scene, table: ObjectDetector, table_tracker: ObjectTracker,
                 belt: ObjectDetector | None = None, belt_tracker: ObjectTracker | None = None,
                 period: float = 0.5, phase_source=None, backend: str | None = None, instance_of=None) -> None:
        self.scene = scene
        self.detectors = {"table": table, **({"belt": belt} if belt is not None else {})}
        self.trackers = {"table": table_tracker, **({"belt": belt_tracker} if belt_tracker is not None else {})}
        self.period = period
        self.phase_source = phase_source      # -> "<arm> <demo phase>" (FrameRecorder.phase)
        self.backend = backend                # pose backend for the 6D estimates (None: none)
        self.instance_of = instance_of        # detection -> the simulator's object name (bookkeeping)
        self.log = DetectionLog()
        self.latest: Look | None = None
        self._renderer: mujoco.Renderer | None = None
        self.look_wall_seconds: list[float] = []
        self.estimate_wall_seconds: list[float] = []
        self.target: dict | None = None       # the object a pick is taking
        self.held: Held | None = None
        # The last whole view of each object: region -> [(centroid xy, label, track)].
        self.identities: dict[str, list[tuple[np.ndarray, str, int]]] = {}
        # 6D estimates: (region, track) -> {pose, local box, t, track position then}.
        self.poses: dict[tuple[str, int], dict] = {}
        self._local_boxes: dict[str, np.ndarray] = {}

    @property
    def renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.scene.model, HEIGHT, WIDTH)
        return self._renderer

    # ---------------------------------------------------------------- 6D poses
    def local_box(self, kind: str) -> np.ndarray:
        """(centre, half) of the object model's box in its own frame (the CAD model the
        pose backend matches, from the registry mesh of that kind)."""
        if kind not in self._local_boxes:
            name = next(n for n, k in self.scene.object_types.items() if k == kind)
            body = int(self.scene.model.joint(self.scene.object_joints[name]).bodyid[0])
            self._local_boxes[kind] = object_box(self.scene.model, body)
        return self._local_boxes[kind]

    def _remember(self, region: str, track: int, kind: str, pose: np.ndarray) -> None:
        tracker = self.trackers.get(region)
        position = tracker.tracks[track].position.copy() if tracker is not None and track in tracker.tracks else pose[:3, 3]
        self.poses[(region, track)] = {"pose": np.asarray(pose, dtype=float).copy(), "box": self.local_box(kind),
                                       "t": float(self.scene.data.time), "position": np.asarray(position, dtype=float)}

    def estimate_6d(self, region: str, detections: list[Detection]) -> int:
        """6D pose of every identified, tracked, not yet estimated detection of `region`,
        in one call of the pose backend. Returns how many were estimated."""
        import time

        from simulation.pick_place.pose_source import get_object_poses

        todo = [d for d in detections if self.backend and d.label is not None and d.track_id is not None
                and not d.occluded and (region, int(d.track_id)) not in self.poses]
        if not todo or self.instance_of is None:
            return 0
        started = time.perf_counter()
        names = [self.instance_of(d) for d in todo]
        self._estimating = True   # these are not the pick target's estimates (on_pose)
        try:
            poses = get_object_poses(self.scene, names, self.backend, masks=[d.mask for d in todo])
        except RuntimeError as error:
            print(f"   (6D estimate failed: {str(error).splitlines()[0]})")
            return 0
        finally:
            self._estimating = False
        for d, pose in zip(todo, poses):
            self._remember(region, int(d.track_id), d.label, pose)
        self.estimate_wall_seconds.append(time.perf_counter() - started)
        print(f"   6D ({self.backend}): {region} " + ", ".join(f"#{d.track_id} {d.label}" for d in todo)
              + f" in {self.estimate_wall_seconds[-1]:.0f}s")
        return len(todo)

    def on_pose(self, scene, key: str, backend: str, pose: np.ndarray, camera: str) -> None:
        """pose_source listener: a pick's own estimate updates its target's 6D pose."""
        if getattr(self, "_estimating", False):
            return
        if self.target is not None and backend == self.backend and scene.object_types.get(key) == self.target["label"]:
            self._remember(self.target["region"], self.target["track"], self.target["label"], pose)

    def pose_now(self, region: str, track: int) -> dict | None:
        """The object's 6D pose now: as estimated (table), or moved on with its track
        (belt: FoundationPose once, the tracker after)."""
        entry = self.poses.get((region, track))
        if entry is None:
            return None
        pose = entry["pose"].copy()
        tracker = self.trackers.get(region)
        if region == "belt" and tracker is not None and track in tracker.tracks:
            pose[:3, 3] += tracker.tracks[track].position - entry["position"]
        return {"pose": pose, "box": entry["box"]}

    # ---------------------------------------------------------------- the tote
    def tote_fill(self, box_xy, box_half, floor_z: float, cell: float = 0.01):
        """From a fresh look into the box: fill(x, y, r) -> the highest point above the
        box floor within r of (x, y) (offsets from the box centre); inf where the
        camera sees less than half of that circle (an arm in the way). The robot's own
        pixels are left out (object_detector self-filter)."""
        look = self.look()
        points = look.frame.points
        ok = np.isfinite(points).all(axis=-1)
        if look.frame.robot is not None:
            ok &= ~look.frame.robot
        p = points[ok]
        dx, dy = p[:, 0] - box_xy[0], p[:, 1] - box_xy[1]
        inside = (np.abs(dx) < box_half[0]) & (np.abs(dy) < box_half[1])
        dx, dy, dz = dx[inside], dy[inside], p[inside, 2] - floor_z
        nx, ny = int(np.ceil(2 * box_half[0] / cell)), int(np.ceil(2 * box_half[1] / cell))
        ix = np.clip(((dx + box_half[0]) / cell).astype(int), 0, nx - 1)
        iy = np.clip(((dy + box_half[1]) / cell).astype(int), 0, ny - 1)
        grid = np.full((nx, ny), -np.inf)
        np.maximum.at(grid, (ix, iy), dz)
        seen = np.isfinite(grid)
        cx = (np.arange(nx) + 0.5) * cell - box_half[0]
        cy = (np.arange(ny) + 0.5) * cell - box_half[1]
        gx, gy = np.meshgrid(cx, cy, indexing="ij")

        def fill(x: float, y: float, r: float) -> float:
            under = (gx - x) ** 2 + (gy - y) ** 2 <= r * r
            if not under.any() or seen[under].mean() < 0.5:
                return np.inf
            return float(grid[under & seen].max())
        return fill

    # ---------------------------------------------------------------- picks
    def set_target(self, side: str, detection: Detection | None) -> None:
        """The object the next pick (by `side`) takes: shown held from its grasp on."""
        self.held = None
        if detection is None or detection.label is None or getattr(detection, "box", None) is None:
            self.target = None
            return
        track = -1 if detection.track_id is None else int(detection.track_id)
        # Moving on the belt: its tracked velocity. A table object stands still -- its
        # tracker's velocity there is look-to-look jitter (a few mm/s; times the wait
        # until the grasp it put a peach's box 5 cm off, 2026-10-01).
        velocity = np.zeros(3)
        belt = self.trackers.get("belt")
        region = "belt" if belt is not None and track in belt.tracks else "table"
        if region == "belt":
            velocity = belt.tracks[track].velocity
        self.target = {"side": side, "label": detection.label, "track": track, "region": region,
                       "box": detection.box.copy(), "t": float(self.scene.data.time),
                       "velocity": np.asarray(velocity, dtype=float)}

    def target_view(self, region: str) -> Detection:
        """A fresh, whole view of the pick target: the identified detection of its kind
        within the tracker gate of where the robot expects it (the box it was chosen
        with, moved on at its tracked velocity). Raises if there is none -- e.g. the
        hand hides it: then the caller keeps its tracked estimate. (Taking the nearest
        detection of the kind at any distance gave a belt pick the other can, 0.65 m
        upstream, while the hand hid the target, 2026-10-01.)"""
        target = self.target
        if target is None:
            raise RuntimeError("perception failed: no pick target")
        look = self.look()
        expected = target["box"].mean(axis=0) + target["velocity"] * (look.t - target["t"])
        gate = self.trackers[region].gate
        near = [(float(np.linalg.norm(d.centroid[:2] - expected[:2])), i, d)
                for i, d in enumerate(look.detections.get(region, [])) if d.label == target["label"]]
        near = [entry for entry in near if entry[0] <= gate]
        if not near:
            raise RuntimeError(f"perception failed: {target['label']} not seen whole within {gate * 100:.0f} cm "
                               "of where it is expected")
        return min(near)[2]

    def _held_now(self, look: Look) -> None:
        """Start, follow or end the held object for this look (from the demo phase)."""
        phase = self.phase_source() if self.phase_source is not None else ""
        side, _, name = (phase or "").partition(" ")
        target = self.target
        if target is None or side != target["side"] or name not in HELD_PHASES:
            self.held = None
            return
        model, data = self.scene.model, self.scene.data
        position, rotation = hand_pose(model, data, side)
        if self.held is None:
            # The grasp starts: the object's 6D pose (else the box it was chosen with,
            # seen whole, the arm away), moved on at its tracked velocity (a belt object
            # travels on while the arm comes; a table object stays), fixed to the hand
            # from here on. The view at the grasp itself is not used: the hand is over
            # the object by then and the camera sees only its top (4-7 cm off for a
            # peach; the chosen box alone was 11-18 cm behind a belt can; 2026-10-01).
            shift = target["velocity"] * (look.t - target["t"])
            estimate = self.poses.get((target["region"], target["track"]))
            if estimate is not None:
                pose = estimate["pose"].copy()
                pose[:3, 3] += target["velocity"] * (look.t - estimate["t"])
                frame = _box_frame(pose, estimate["box"])
                self.held = Held(side, target["label"], target["track"], rotation.T @ (frame[:3, 3] - position),
                                 rotation.T @ frame[:3, :3], estimate["box"][3:].copy(), six_d=True)
            else:
                box = target["box"] + shift
                centre, half = box.mean(axis=0), 0.5 * (box[1] - box[0])
                self.held = Held(side, target["label"], target["track"], rotation.T @ (centre - position),
                                 rotation.T.copy(), half)
        look.held.append((self.held, self.held.box(position, rotation)))

    # ---------------------------------------------------------------- looking
    def look(self, max_age: float = 0.0) -> Look:
        """A look no older than `max_age` s of sim time (a new one if needed). Only
        identified detections are kept (label set)."""
        import time

        from simulation.fp_bridge import camera_pose_cv, intrinsics

        t = float(self.scene.data.time)
        if self.latest is not None and t - self.latest.t <= max_age:
            return self.latest
        started = time.perf_counter()
        first = next(iter(self.detectors.values()))
        frame = first.frame(self.scene.data, self.renderer)
        look = Look(t, frame)
        for region, detector in self.detectors.items():
            if not detector.references:
                continue  # not enrolled yet
            found = detector.detect(self.scene.data, unique=False, frame=frame)
            # Partly hidden by the robot: no label of its own and no tracker update (the
            # view is not the object's); shown as the object last seen whole there.
            detections = [d for d in found if d.label is not None and not d.occluded]
            for d in found:
                d.box = detection_box(frame, d)
            self.trackers[region].update(detections, t)
            look.detections[region] = detections
            gate = self.trackers[region].gate
            for d in (d for d in found if d.occluded):
                known = [(float(np.linalg.norm(c - d.centroid[:2])), label, track)
                         for c, label, track in self.identities.get(region, [])]
                if known and min(known)[0] <= gate:
                    _, label, track = min(known)
                    look.occluded.setdefault(region, []).append((d, label, track))
            if detections:
                seen = [(d.centroid[:2].copy(), d.label, -1 if d.track_id is None else int(d.track_id)) for d in detections]
                hidden = [entry for entry in self.identities.get(region, [])
                          if all(float(np.linalg.norm(entry[0] - s[0])) > gate for s in seen)]
                # Whole views now, plus the identities of objects not seen whole this time.
                self.identities[region] = seen + hidden
        self._held_now(look)
        model, data = self.scene.model, self.scene.data
        self.log.add(look, self._shown(look), intrinsics(model, first.camera), camera_pose_cv(data, model, first.camera))
        self.latest = look
        self.look_wall_seconds.append(time.perf_counter() - started)
        return look

    def _shown(self, look: Look) -> list[Shown]:
        held_boxes = [box for _, box in look.held]
        shown: list[Shown] = []

        def add(region: str, d: Detection, label: str, track: int, cost: float, velocity: np.ndarray) -> None:
            if any(_overlaps_xy(d.box, b) for b in held_boxes):
                return
            rows, cols = np.nonzero(d.mask)
            item = Shown(region, label, track, cost, d.box, np.asarray(d.centroid, dtype=float), velocity,
                         rect=(int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())))
            estimate = self.pose_now(region, track) if track >= 0 else None
            if estimate is not None:
                item.frame = _box_frame(estimate["pose"], estimate["box"])
                item.half = estimate["box"][3:].copy()
            shown.append(item)

        for region, detections in look.detections.items():
            tracker = self.trackers.get(region)
            for d in detections:
                track = -1 if d.track_id is None else int(d.track_id)
                state = tracker.tracks.get(track) if tracker is not None else None
                velocity = state.velocity if state is not None and region == "belt" else np.zeros(3)
                add(region, d, d.label, track, float(min(d.cost, 1e6)), np.asarray(velocity, dtype=float))
        for region, entries in look.occluded.items():
            for d, label, track in entries:
                add(region, d, label, track, float("nan"), np.zeros(3))
        model, data = self.scene.model, self.scene.data
        for held, box in look.held:
            position, rotation = hand_pose(model, data, held.side)
            frame = held.frame(position, rotation)
            anchor = np.concatenate([[SIDES.index(held.side)], held.offset, held.rotation.ravel(), held.half,
                                     [1.0 if held.six_d else 0.0]])
            shown.append(Shown("held", held.label, held.track, 0.0, box, box.mean(axis=0), np.zeros(3),
                               frame=frame if held.six_d else None, half=held.half if held.six_d else None,
                               anchor=anchor))
        return shown

    def on_frame(self, t: float) -> None:
        """FrameRecorder hook: look once every `period` s of sim time."""
        if self.latest is not None and t - self.latest.t < self.period:
            return
        if not any(d.references for d in self.detectors.values()):
            return
        try:
            self.look()
        except RuntimeError as error:  # e.g. the workspace hidden behind an arm: skip this look
            print(f"   (look skipped at t={t:.1f}s: {str(error).splitlines()[0]})")
            self.latest = Look(t, self.latest.frame if self.latest else None)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class DetectionReplayOverlay:
    """Replay: the latest look's objects (6D model boxes with x/y/z axes, or tracked
    boxes), the held object recomputed from the hand in every frame (bind() gives the
    replayed model/data), and the look's annotated camera image. A look is shown until
    the next one, at most 2 x the look period measured from the recording."""

    def __init__(self, recording) -> None:
        self.times = recording["det_look_times"] if "det_look_times" in recording.files else np.zeros(0)
        if len(self.times):
            n = len(recording["det_look"])
            self.look = recording["det_look"]
            self.region, self.label, self.track = recording["det_region"], recording["det_label"], recording["det_track"]
            self.box, self.centroid, self.velocity = recording["det_box"], recording["det_centroid"], recording["det_velocity"]
            self.frame = recording["det_frame"] if "det_frame" in recording.files else np.full((n, 16), np.nan)
            self.half = recording["det_half"] if "det_half" in recording.files else np.full((n, 3), np.nan)
            anchor = recording["det_anchor"] if "det_anchor" in recording.files else np.full((n, 17), np.nan)
            if anchor.shape[1] == 16:   # recordings from before the 6D flag
                anchor = np.hstack([anchor, np.zeros((n, 1))])
            self.anchor = anchor
            self.images = recording["det_images"]
        self.max_age = 2.0 * float(np.median(np.diff(self.times))) if len(self.times) > 1 else 0.0
        self.model = self.data = None

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model, self.data = model, data

    def current(self, t: float) -> int:
        index = int(np.searchsorted(self.times, t, side="right")) - 1
        if index < 0 or t - float(self.times[index]) > self.max_age:
            return -1
        return index

    def _geometry(self, row: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """(axis-aligned box, 6D frame or None, half or None) for this row now."""
        anchor = self.anchor[row]
        if self.model is not None and np.isfinite(anchor[0]):
            held = Held(SIDES[int(anchor[0])], str(self.label[row]), int(self.track[row]), anchor[1:4],
                        anchor[4:13].reshape(3, 3), anchor[13:16], six_d=bool(anchor[16] > 0.5))
            position, rotation = hand_pose(self.model, self.data, held.side)
            frame = held.frame(position, rotation)
            return held.box(position, rotation), (frame if held.six_d else None), (held.half if held.six_d else None)
        if np.isfinite(self.frame[row][0]):
            return self.box[row], self.frame[row].reshape(4, 4), self.half[row]
        return self.box[row], None, None

    def markers(self, scn: mujoco.MjvScene, t: float) -> None:
        index = self.current(t)
        if index < 0:
            return
        for row in np.flatnonzero(self.look == index):
            region = str(self.region[row])
            rgba = REGION_RGBA.get(region, REGION_RGBA["table"])
            box, frame, half = self._geometry(row)
            if frame is not None:
                corners = _corners(frame, half)
                for a, b in BOX_EDGES:
                    _connector(scn, mujoco.mjtGeom.mjGEOM_LINE, REPLAY_LINE_WIDTH_PX, corners[a], corners[b], rgba)
                for k in range(3):
                    _connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, REPLAY_AXIS_WIDTH_M, frame[:3, 3],
                               frame[:3, 3] + frame[:3, k] * REPLAY_AXIS_LENGTH_M, AXIS_RGBA[k])
                top = float(corners[:, 2].max())
                centre = frame[:3, 3]
            else:
                low, high = box
                corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
                for a, b in BOX_EDGES:
                    _connector(scn, mujoco.mjtGeom.mjGEOM_LINE, 2.0, corners[a], corners[b], rgba)
                top = float(high[2])
                centre = 0.5 * (low + high)
            name = (f"#{self.track[row]} " if self.track[row] >= 0 else "") + str(self.label[row]) + (" 6D" if frame is not None else "")
            _label(scn, np.array([centre[0], centre[1], top + 0.03]), name, rgba)
            v = self.velocity[row]
            if np.linalg.norm(v) > 0.0:
                start = self.centroid[row]
                _connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.004, start, start + v * VELOCITY_ARROW_S, rgba)

    def image(self, t: float):
        index = self.current(t)
        return None if index < 0 else ("det", index, self.images[index])


def _connector(scn, kind, width, a, b, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, kind, width, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    scn.ngeom += 1


def _label(scn, position, text: str, rgba) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.004), np.asarray(position, dtype=float),
                        np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
    geom.label = text
    scn.ngeom += 1
