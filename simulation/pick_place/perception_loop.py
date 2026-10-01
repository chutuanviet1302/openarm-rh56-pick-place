"""Continuous perception for a task: the head camera looks every `period` s of sim time,
the table and the belt detectors share that one RGB-D frame, each feeds its tracker,
and every look is logged (all detections, not only the object being picked) so the
replay can show what the robot saw.

    perception = Perception(scene, table_detector, table_tracker, belt_detector, belt_tracker)
    recorder.hooks.append(perception.on_frame)     # look while the task runs
    look = perception.look()                        # a fresh look for a decision
    recorder.extra.update(perception.log.arrays())  # saved with the replay frames

Replay: DetectionReplayOverlay draws, for the latest look, a box around each detection
(world axis-aligned, from its depth points) labelled "#id kind", the belt velocity as
an arrow, and the annotated camera image as an inset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from simulation.fp_bridge import HEIGHT, WIDTH
from simulation.object_detector import CameraFrame, Detection, ObjectDetector, ObjectTracker
from simulation.pick_place.pose_overlay import INSET_SIZE

REGIONS = ("table", "belt")
# Image colours (RGB) and marker colours (RGBA) per region; unidentified: grey.
REGION_RGB = {"table": (40, 200, 255), "belt": (255, 170, 0), "unknown": (170, 170, 170)}
REGION_RGBA = {"table": (0.15, 0.8, 1.0, 1.0), "belt": (1.0, 0.65, 0.0, 1.0), "unknown": (0.7, 0.7, 0.7, 1.0)}
# Box extent from the detection's depth points: these percentiles, not min/max, so the
# few mixed-depth pixels on a silhouette edge do not stretch the box.
EXTENT_PERCENTILES = (1.0, 99.0)
VELOCITY_ARROW_S = 2.0      # the belt arrow shows where the object will be this much later


@dataclass
class Look:
    t: float
    frame: CameraFrame
    detections: dict[str, list[Detection]] = field(default_factory=dict)


class DetectionLog:
    """Every look's detections, flattened for np.savez (one row per detection)."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.rows: list[tuple] = []           # (look index, region, label, track, cost)
        self.boxes: list[np.ndarray] = []     # (2, 3) world min / max
        self.centroids: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.images: list[np.ndarray] = []

    def add(self, look: Look, trackers: dict[str, ObjectTracker], camera_K, T_world_cam) -> None:
        import cv2

        index = len(self.times)
        self.times.append(look.t)
        image = cv2.cvtColor(look.frame.rgb, cv2.COLOR_RGB2BGR)
        for region, detections in look.detections.items():
            tracker = trackers.get(region)
            for d in detections:
                points = look.frame.points[d.mask]
                points = points[np.isfinite(points).all(axis=1)]
                box = np.percentile(points, EXTENT_PERCENTILES, axis=0)
                track = tracker.tracks.get(d.track_id) if tracker is not None and d.track_id is not None else None
                velocity = track.velocity if track is not None and region == "belt" else np.zeros(3)
                self.rows.append((index, region, d.label or "unknown", -1 if d.track_id is None else int(d.track_id),
                                  float(min(d.cost, 1e6))))
                self.boxes.append(box)
                self.centroids.append(np.asarray(d.centroid, dtype=float))
                self.velocities.append(np.asarray(velocity, dtype=float))
                colour = REGION_RGB[region if d.label else "unknown"][::-1]
                rows, cols = np.nonzero(d.mask)
                cv2.rectangle(image, (int(cols.min()), int(rows.min())), (int(cols.max()), int(rows.max())), colour, 2)
                text = f"#{d.track_id} {d.label}" if d.label else "unknown"
                cv2.putText(image, text, (int(cols.min()), max(12, int(rows.min()) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, text, (int(cols.min()), max(12, int(rows.min()) - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colour, 1, cv2.LINE_AA)
                if np.linalg.norm(velocity) > 0.0:
                    ends = np.vstack([d.centroid, d.centroid + velocity * VELOCITY_ARROW_S])
                    uv = _project(ends, camera_K, T_world_cam).astype(int)
                    cv2.arrowedLine(image, tuple(uv[0]), tuple(uv[1]), colour, 2, cv2.LINE_AA, tipLength=0.25)
        counts = {r: len(look.detections.get(r, [])) for r in REGIONS}
        caption = f"t={look.t:5.1f}s  table {counts['table']}  belt {counts['belt']}"
        cv2.putText(image, caption, (8, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, caption, (8, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
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
            det_images=np.asarray(self.images, dtype=np.uint8).reshape(len(self.images), INSET_SIZE[1], INSET_SIZE[0], 3),
        )


def _project(points: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    world_to_cam = np.linalg.inv(T_world_cam)
    cam = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    uv = cam @ K.T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)


class Perception:
    """One camera, up to two detectors (table, belt) with their trackers."""

    def __init__(self, scene, table: ObjectDetector, table_tracker: ObjectTracker,
                 belt: ObjectDetector | None = None, belt_tracker: ObjectTracker | None = None,
                 period: float = 0.5) -> None:
        self.scene = scene
        self.detectors = {"table": table, **({"belt": belt} if belt is not None else {})}
        self.trackers = {"table": table_tracker, **({"belt": belt_tracker} if belt_tracker is not None else {})}
        self.period = period
        self.log = DetectionLog()
        self.latest: Look | None = None
        self._renderer: mujoco.Renderer | None = None
        self.look_wall_seconds: list[float] = []

    @property
    def renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.scene.model, HEIGHT, WIDTH)
        return self._renderer

    def look(self, max_age: float = 0.0) -> Look:
        """A look no older than `max_age` s of sim time (a new one if needed)."""
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
            detections = detector.detect(self.scene.data, unique=False, frame=frame)
            self.trackers[region].update(detections, t)
            look.detections[region] = detections
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
    """Replay: the latest look's detections as labelled 3-D boxes (+ belt arrows) and
    its annotated camera image. A look is shown until the next one, at most
    2 x the look period measured from the recording (a gap means no look)."""

    def __init__(self, recording) -> None:
        self.times = recording["det_look_times"] if "det_look_times" in recording.files else np.zeros(0)
        if len(self.times):
            self.look = recording["det_look"]
            self.region, self.label, self.track = recording["det_region"], recording["det_label"], recording["det_track"]
            self.box, self.centroid, self.velocity = recording["det_box"], recording["det_centroid"], recording["det_velocity"]
            self.images = recording["det_images"]
        self.max_age = 2.0 * float(np.median(np.diff(self.times))) if len(self.times) > 1 else 0.0

    def current(self, t: float) -> int:
        index = int(np.searchsorted(self.times, t, side="right")) - 1
        if index < 0 or t - float(self.times[index]) > self.max_age:
            return -1
        return index

    def markers(self, scn: mujoco.MjvScene, t: float) -> None:
        index = self.current(t)
        if index < 0:
            return
        for row in np.flatnonzero(self.look == index):
            kind = str(self.region[row]) if str(self.label[row]) != "unknown" else "unknown"
            rgba = REGION_RGBA[kind]
            low, high = self.box[row]
            corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
            from simulation.pick_place.pose_overlay import BOX_EDGES

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
