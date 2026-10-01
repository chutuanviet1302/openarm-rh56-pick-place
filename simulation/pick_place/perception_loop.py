"""Continuous perception for a task: the head camera looks every `period` s of sim time,
the table and the belt detectors share that one RGB-D frame, each feeds its tracker,
and every look is logged (all identified objects, not only the one being picked) so
the replay can show what the robot saw.

    perception = Perception(scene, table_detector, table_tracker, belt_detector, belt_tracker,
                            phase_source=lambda: recorder.phase)
    recorder.hooks.append(perception.on_frame)     # look while the task runs
    look = perception.look()                        # a fresh look for a decision
    perception.set_target("right", detection)       # the object a pick is about to take
    recorder.extra.update(perception.log.arrays())  # saved with the replay frames

What is shown (and logged):
    identified objects   the camera's detections with a label (the robot's own arm and
                         hand pixels are removed first, object_detector self-filter);
                         candidates that match no known object are not shown
    the held object      from the grasp until the drop (demo phases "grasp", "carry") the
                         object being picked is drawn where the robot holds it: its box,
                         taken from the camera when the grasp starts, fixed to the hand
                         (hand pose from the arm's joint angles, forward kinematics) --
                         the camera no longer sees it whole between the fingers, or at
                         all once it is lifted off the table. Detections overlapping it
                         are left out of the display (partial views between the fingers);
                         the trackers still get every detection.

Replay: DetectionReplayOverlay draws, for the latest look, a box around each object
(world axis-aligned, from its depth points) labelled "#id kind", the belt velocity as
an arrow, the held object's box recomputed from the hand in every frame, and the
annotated camera image as an inset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from simulation.fp_bridge import HEIGHT, WIDTH
from simulation.object_detector import CameraFrame, Detection, ObjectDetector, ObjectTracker
from simulation.pick_place.pose_overlay import INSET_SIZE

REGIONS = ("table", "belt")
HELD_PHASES = ("grasp", "carry")
SIDES = ("left", "right")
# Image colours (RGB) and marker colours (RGBA) per region.
REGION_RGB = {"table": (40, 200, 255), "belt": (255, 170, 0), "held": (60, 230, 90)}
REGION_RGBA = {"table": (0.15, 0.8, 1.0, 1.0), "belt": (1.0, 0.65, 0.0, 1.0), "held": (0.25, 0.95, 0.35, 1.0)}
# Box extent from the detection's depth points: these percentiles, not min/max, so the
# few mixed-depth pixels on a silhouette edge do not stretch the box.
EXTENT_PERCENTILES = (1.0, 99.0)
VELOCITY_ARROW_S = 2.0      # the belt arrow shows where the object will be this much later
_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)


@dataclass
class Held:
    """The object in a hand: its box half-size and pose in the hand (EE site) frame."""
    side: str
    label: str
    track: int
    offset: np.ndarray        # box centre in the hand frame
    rotation: np.ndarray      # box axes in the hand frame (3x3)
    half: np.ndarray          # box half-size along its axes

    def box(self, hand_position: np.ndarray, hand_rotation: np.ndarray) -> np.ndarray:
        """World axis-aligned (2, 3) min / max of the held box for this hand pose."""
        centre = hand_position + hand_rotation @ self.offset
        corners = centre + (_SIGNS * self.half) @ (hand_rotation @ self.rotation).T
        return np.stack([corners.min(axis=0), corners.max(axis=0)])


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


class DetectionLog:
    """Every look's shown objects, flattened for np.savez (one row per object)."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.rows: list[tuple] = []           # (look index, region, label, track, cost)
        self.boxes: list[np.ndarray] = []     # (2, 3) world min / max
        self.centroids: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.anchors: list[np.ndarray] = []   # held rows: side, offset(3), rotation(9), half(3); else NaN
        self.images: list[np.ndarray] = []

    def add(self, look: Look, trackers: dict[str, ObjectTracker], camera_K, T_world_cam) -> None:
        import cv2

        index = len(self.times)
        self.times.append(look.t)
        image = cv2.cvtColor(look.frame.rgb, cv2.COLOR_RGB2BGR)
        held_boxes = [box for _, box in look.held]
        shown: list[tuple] = []  # (region, label, track, cost, box, centroid, velocity, anchor, pixel rect)
        for region, detections in look.detections.items():
            tracker = trackers.get(region)
            for d in detections:
                if d.label is None or any(_overlaps_xy(d.box, b) for b in held_boxes):
                    continue
                track = tracker.tracks.get(d.track_id) if tracker is not None and d.track_id is not None else None
                velocity = track.velocity if track is not None and region == "belt" else np.zeros(3)
                rows, cols = np.nonzero(d.mask)
                rect = (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
                shown.append((region, d.label, -1 if d.track_id is None else int(d.track_id), float(min(d.cost, 1e6)),
                              d.box, np.asarray(d.centroid, dtype=float), np.asarray(velocity, dtype=float), None, rect))
        for region, entries in look.occluded.items():
            for d, label, track in entries:
                if any(_overlaps_xy(d.box, b) for b in held_boxes):
                    continue
                rows, cols = np.nonzero(d.mask)
                rect = (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
                shown.append((region, label, track, float("nan"), d.box, np.asarray(d.centroid, dtype=float), np.zeros(3),
                              None, rect))
        for held, box in look.held:
            uv = _project(box[0] + (_SIGNS * 0.5 + 0.5) * (box[1] - box[0]), camera_K, T_world_cam)
            rect = (int(uv[:, 0].min()), int(uv[:, 1].min()), int(uv[:, 0].max()), int(uv[:, 1].max()))
            anchor = np.concatenate([[SIDES.index(held.side)], held.offset, held.rotation.ravel(), held.half])
            shown.append(("held", held.label, held.track, 0.0, box, box.mean(axis=0), np.zeros(3), anchor, rect))
        for region, label, track, cost, box, centroid, velocity, anchor, rect in shown:
            self.rows.append((index, region, label, track, cost))
            self.boxes.append(box)
            self.centroids.append(centroid)
            self.velocities.append(velocity)
            self.anchors.append(anchor if anchor is not None else np.full(16, np.nan))
            colour = REGION_RGB[region][::-1]
            cv2.rectangle(image, rect[:2], rect[2:], colour, 2)
            text = f"#{track} {label}" if track >= 0 else label
            for thick, ink in ((3, (0, 0, 0)), (1, colour)):
                cv2.putText(image, text, (rect[0], max(12, rect[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ink, thick,
                            cv2.LINE_AA)
            if np.linalg.norm(velocity) > 0.0:
                ends = np.vstack([centroid, centroid + velocity * VELOCITY_ARROW_S])
                uv = _project(ends, camera_K, T_world_cam).astype(int)
                cv2.arrowedLine(image, tuple(uv[0]), tuple(uv[1]), colour, 2, cv2.LINE_AA, tipLength=0.25)
        counts = {r: sum(1 for s in shown if s[0] == r) for r in (*REGIONS, "held")}
        caption = f"t={look.t:5.1f}s  table {counts['table']}  belt {counts['belt']}  held {counts['held']}"
        for thick, ink in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            cv2.putText(image, caption, (8, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, thick, cv2.LINE_AA)
        image = cv2.resize(image, INSET_SIZE, interpolation=cv2.INTER_AREA)
        self.images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self.rows)
        return dict(
            det_look_times=np.asarray(self.times, dtype=float),
            det_look=np.asarray([r[0] for r in self.rows], dtype=int),
            det_region=np.asarray([r[1] for r in self.rows], dtype=str),
            det_label=np.asarray([r[2] for r in self.rows], dtype=str),
            det_track=np.asarray([r[3] for r in self.rows], dtype=int),
            det_cost=np.asarray([r[4] for r in self.rows], dtype=float),
            det_box=np.asarray(self.boxes, dtype=float).reshape(n, 2, 3),
            det_centroid=np.asarray(self.centroids, dtype=float).reshape(n, 3),
            det_velocity=np.asarray(self.velocities, dtype=float).reshape(n, 3),
            det_anchor=np.asarray(self.anchors, dtype=float).reshape(n, 16),
            det_images=np.asarray(self.images, dtype=np.uint8).reshape(len(self.images), INSET_SIZE[1], INSET_SIZE[0], 3),
        )


def _project(points: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    world_to_cam = np.linalg.inv(T_world_cam)
    cam = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    uv = cam @ K.T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)


def hand_pose(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> tuple[np.ndarray, np.ndarray]:
    """The hand (EE site) pose from the arm's current kinematics."""
    from simulation.pick_place.config import EE_SITE

    site = model.site(EE_SITE[side]).id
    return data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3).copy()


class Perception:
    """One camera, up to two detectors (table, belt) with their trackers."""

    def __init__(self, scene, table: ObjectDetector, table_tracker: ObjectTracker,
                 belt: ObjectDetector | None = None, belt_tracker: ObjectTracker | None = None,
                 period: float = 0.5, phase_source=None) -> None:
        self.scene = scene
        self.detectors = {"table": table, **({"belt": belt} if belt is not None else {})}
        self.trackers = {"table": table_tracker, **({"belt": belt_tracker} if belt_tracker is not None else {})}
        self.period = period
        self.phase_source = phase_source      # -> "<arm> <demo phase>" (FrameRecorder.phase)
        self.log = DetectionLog()
        self.latest: Look | None = None
        self._renderer: mujoco.Renderer | None = None
        self.look_wall_seconds: list[float] = []
        self.target: dict | None = None       # the object a pick is taking
        self.held: Held | None = None
        # The last whole view of each object: region -> [(centroid xy, label, track)].
        self.identities: dict[str, list[tuple[np.ndarray, str, int]]] = {}

    @property
    def renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.scene.model, HEIGHT, WIDTH)
        return self._renderer

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
        if belt is not None and track in belt.tracks:
            velocity = belt.tracks[track].velocity
        self.target = {"side": side, "label": detection.label, "track": track, "box": detection.box.copy(),
                       "t": float(self.scene.data.time), "velocity": np.asarray(velocity, dtype=float)}

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
            # The grasp starts: the box the object was chosen with (seen whole, the arm
            # away), moved on at its tracked velocity (a belt object travels on while
            # the arm comes; a table object stays), fixed to the hand from here on. The
            # view at the grasp itself is not used: the hand is over the object by then
            # and the camera sees only its top (4-7 cm off for a peach; the chosen box
            # alone was 11-18 cm behind a belt can; 2026-10-01).
            box = target["box"] + target["velocity"] * (look.t - target["t"])
            centre, half = box.mean(axis=0), 0.5 * (box[1] - box[0])
            self.held = Held(side, target["label"], target["track"], rotation.T @ (centre - position), rotation.T.copy(), half)
        look.held.append((self.held, self.held.box(position, rotation)))

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
        self.log.add(look, self.trackers, intrinsics(model, first.camera), camera_pose_cv(data, model, first.camera))
        self.latest = look
        self.look_wall_seconds.append(time.perf_counter() - started)
        return look

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
    """Replay: the latest look's objects as labelled 3-D boxes (+ belt arrows), the held
    object's box recomputed from the hand in every frame (bind() gives the replayed
    model/data), and the look's annotated camera image. A look is shown until the next
    one, at most 2 x the look period measured from the recording (a gap means no look)."""

    def __init__(self, recording) -> None:
        self.times = recording["det_look_times"] if "det_look_times" in recording.files else np.zeros(0)
        if len(self.times):
            self.look = recording["det_look"]
            self.region, self.label, self.track = recording["det_region"], recording["det_label"], recording["det_track"]
            self.box, self.centroid, self.velocity = recording["det_box"], recording["det_centroid"], recording["det_velocity"]
            self.anchor = recording["det_anchor"] if "det_anchor" in recording.files else np.full((len(self.look), 16), np.nan)
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

    def _box(self, row: int) -> np.ndarray:
        anchor = self.anchor[row]
        if self.model is None or not np.isfinite(anchor[0]):
            return self.box[row]
        held = Held(SIDES[int(anchor[0])], str(self.label[row]), int(self.track[row]), anchor[1:4], anchor[4:13].reshape(3, 3),
                    anchor[13:16])
        return held.box(*hand_pose(self.model, self.data, held.side))

    def markers(self, scn: mujoco.MjvScene, t: float) -> None:
        index = self.current(t)
        if index < 0:
            return
        from simulation.pick_place.pose_overlay import BOX_EDGES

        for row in np.flatnonzero(self.look == index):
            region = str(self.region[row])
            rgba = REGION_RGBA.get(region, REGION_RGBA["table"])
            low, high = self._box(row)
            corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
            for a, b in BOX_EDGES:
                _connector(scn, mujoco.mjtGeom.mjGEOM_LINE, 2.0, corners[a], corners[b], rgba)
            name = f"#{self.track[row]} {self.label[row]}" if self.track[row] >= 0 else str(self.label[row])
            _label(scn, (low + high) / 2.0 + np.array([0.0, 0.0, (high[2] - low[2]) / 2.0 + 0.03]), name, rgba)
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
