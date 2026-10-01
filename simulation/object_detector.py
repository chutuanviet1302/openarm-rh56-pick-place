"""Find, identify and track the task objects from the head camera's RGB-D alone.

No simulator state is read: every detection comes from the rendered colour and
depth image, as it would from the D435 on the robot.

    detector = ObjectDetector(model, known=("can", "tuna_can", "gelatin_box", "orange"))
    detector.enroll(scene_factory)         # reference features per object (once)
    detections = detector.detect(scene)    # label, pixel mask, 3D centroid per object
    tracker.update(detections)             # stable ids across looks

Pipeline:
    1. depth -> points in the world frame (camera K and pose; simulation/fp_bridge)
    2. the work surface = the dominant height of points in the workspace
    3. object pixels = points above the surface, inside the workspace, outside the
       basket (objects already in the basket are done)
    4. connected components in the image -> one candidate per object
    5. features per candidate: height above the surface, footprint (minimum-area
       rectangle), fill (convex hull / rectangle: a disc 0.79, a box 1.0), mean hue
    6. identity: assignment to the enrolled references (Hungarian, each object at most
       once), rejected above a cost threshold

Enrollment renders each known object alone once (its reference features) -- the sim
stand-in for photographing each object once on the real robot.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

from simulation.fp_bridge import HEIGHT, WIDTH, camera_pose_cv, intrinsics

MIN_OBJECT_PIXELS = 150
ABOVE_SURFACE_M = 0.008
HEIGHT_STEP_M = 0.008       # neighbouring pixels further apart in height: an object edge
DEPTH_STEP_M = 0.01         # neighbouring pixels further apart in range: an occlusion edge
# Feature scales for the identity cost: a difference of one scale unit costs 1.
SCALES = {"height": 0.012, "short": 0.012, "long": 0.012, "fill": 0.08, "hue": 0.06}
MAX_COST = 12.0
# Robot self-filter: the robot's own arm and hand pixels are removed before looking for
# objects (it knows its geometry and joint angles -- as robot_self_filter does with a
# real depth camera). Without it a hand over the table was an "unknown" object and a
# can in a closing hand, merged with the fingers, was taken for an orange (2026-10-01).
# The mask is grown by this many pixels: depth at a silhouette edge mixes hand and
# whatever is behind it.
ROBOT_MASK_GROW_PX = 2
ROBOT_BODY_PREFIXES = ("openarm_left_link", "openarm_right_link", "inspire_")
ROBOT_GEOM_GROUP = 5                # the robot-only render's geom group
ROBOT_DEPTH_TOLERANCE_M = 0.005     # robot depth within this of the scene's: the robot is what is seen


@dataclass
class Features:
    height: float      # m above the work surface (top of the object)
    short: float       # m, footprint minimum-area rectangle, short side
    long: float        # m, long side
    fill: float        # convex hull area / rectangle area
    hue: float         # 0..1, circular
    saturation: float

    def as_dict(self) -> dict:
        return {k: round(float(v), 4) for k, v in self.__dict__.items()}


@dataclass
class CameraFrame:
    rgb: np.ndarray                   # (H, W, 3) uint8
    depth: np.ndarray                 # (H, W) m along the optical axis
    points: np.ndarray                # (H, W, 3) world, NaN where there is no depth
    robot: np.ndarray | None = None   # (H, W) bool: pixels of the robot's own arms and hands


@dataclass
class Detection:
    label: str | None
    cost: float
    mask: np.ndarray                  # (H, W) bool
    centroid: np.ndarray              # world, m (mean of the visible points)
    features: Features
    pixels: int
    track_id: int | None = None
    # Touches the robot's own pixels: partly hidden by an arm or hand, so its visible
    # shape and colour do not describe the whole object (2.4% of the labels taken from
    # such views were wrong, full conveyor run 2026-10-01).
    occluded: bool = False


def _hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def _min_area_rectangle(xy: np.ndarray) -> tuple[float, float, float]:
    """(short side, long side, angle rad) of the minimum-area enclosing rectangle."""
    best = None
    for angle in np.radians(np.arange(0.0, 90.0, 1.0)):
        c, s = np.cos(angle), np.sin(angle)
        rotated = xy @ np.array([[c, -s], [s, c]])
        extent = np.ptp(rotated, axis=0)
        area = float(extent[0] * extent[1])
        if best is None or area < best[0]:
            best = (area, float(extent.min()), float(extent.max()), float(angle))
    return best[1], best[2], best[3]


class ObjectDetector:
    def __init__(
        self, model: mujoco.MjModel, known, camera: str = "d435_head", *,
        workspace_x=(0.17, 0.67), workspace_y=(-0.55, 0.55), basket_xy=None, basket_margin: float = 0.12,
    ) -> None:
        self.model = model
        # Each entry: a registry key, or (key, rest pose) -- one reference per pose, so
        # a lying can (71 mm tall, rectangular footprint) is still "can".
        self.known = tuple(k if isinstance(k, tuple) else (k, "upright") for k in known)
        self.camera = camera
        self.workspace_x, self.workspace_y = workspace_x, workspace_y
        self.basket_xy = None if basket_xy is None else np.asarray(basket_xy, dtype=float)
        self.basket_margin = basket_margin
        self.references: dict[str, list[Features]] = {}

    # ------------------------------------------------------------------ rendering
    def _render(self, data: mujoco.MjData, renderer: mujoco.Renderer) -> tuple[np.ndarray, np.ndarray]:
        # `renderer` must belong to the same model as `data`.
        # No shadow or reflection passes: they are 78% of the frame time (673 -> 152 ms
        # per 640x480 frame on the laptop's Intel Iris Xe, shadowsize 8192, 2026-10-01)
        # and depth never needs them. Enrollment renders through here too, so the
        # reference colours are taken under the same lighting as every look.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        renderer.update_scene(data, camera=self.camera)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=self.camera)
        depth = renderer.render().astype(np.float64)
        renderer.disable_depth_rendering()
        return rgb, depth

    def _points(self, model: mujoco.MjModel, data: mujoco.MjData, depth: np.ndarray) -> np.ndarray:
        """(H, W, 3) world points for every pixel (NaN where there is no depth)."""
        K = intrinsics(model, self.camera, WIDTH, HEIGHT)
        T = camera_pose_cv(data, model, self.camera)
        v, u = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
        z = depth.copy()
        far = float(model.vis.map.zfar * model.stat.extent)
        z[(z <= 0) | (z >= 0.99 * far)] = np.nan
        cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=-1)
        return cam @ T[:3, :3].T + T[:3, 3]

    # ------------------------------------------------------------------ detection
    @staticmethod
    def robot_geoms(model: mujoco.MjModel) -> np.ndarray:
        """Geom ids on the robot's arms and hands (body or an ancestor named so)."""
        ids = []
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            while body:
                if (model.body(body).name or "").startswith(ROBOT_BODY_PREFIXES):
                    ids.append(geom)
                    break
                body = int(model.body_parentid[body])
        return np.asarray(ids, dtype=int)

    def _robot_mask(self, model: mujoco.MjModel, data: mujoco.MjData, depth: np.ndarray) -> np.ndarray | None:
        """Pixels of the robot's own arms and hands: its model alone rendered as depth at
        the current joint angles (a copy of the model with only the arm and hand geoms
        in a group of their own), where that depth is not behind the scene's. Only for
        the detector's own model (enrollment scenes keep the arms out of the way)."""
        if model is not self.model:
            return None
        if self.__dict__.get("_robot_view") is None:
            import copy

            robot_model = copy.copy(model)
            robot_model.geom_group[:] = 0
            robot_model.geom_group[self.robot_geoms(model)] = ROBOT_GEOM_GROUP
            option = mujoco.MjvOption()
            option.geomgroup[:] = 0
            option.geomgroup[ROBOT_GEOM_GROUP] = 1
            renderer = mujoco.Renderer(robot_model, HEIGHT, WIDTH)
            renderer.enable_depth_rendering()
            self._robot_view = (renderer, option)
        renderer, option = self._robot_view
        renderer.update_scene(data, camera=self.camera, scene_option=option)
        robot_depth = renderer.render().astype(np.float64)
        far = float(model.vis.map.zfar * model.stat.extent)
        robot = (robot_depth > 0) & (robot_depth < 0.99 * far) & (robot_depth <= depth + ROBOT_DEPTH_TOLERANCE_M)
        if ROBOT_MASK_GROW_PX:
            robot = ndimage.binary_dilation(robot, iterations=ROBOT_MASK_GROW_PX)
        return robot

    def frame(self, data: mujoco.MjData, renderer: mujoco.Renderer | None = None,
              model: mujoco.MjModel | None = None) -> "CameraFrame":
        """One RGB-D frame of this detector's camera as world points, with the robot's
        own pixels marked; several detectors on the same camera can share it
        (candidates(frame=...))."""
        model = model or self.model
        own = renderer is None
        renderer = renderer or mujoco.Renderer(model, HEIGHT, WIDTH)
        try:
            rgb, depth = self._render(data, renderer)
        finally:
            if own:
                renderer.close()
        return CameraFrame(rgb, depth, self._points(model, data, depth), self._robot_mask(model, data, depth))

    def candidates(self, data: mujoco.MjData, renderer: mujoco.Renderer | None = None,
                   model: mujoco.MjModel | None = None, frame: "CameraFrame | None" = None) -> list[Detection]:
        """Unlabelled object candidates in the current frame. `model` defaults to the
        detector's own; pass the scene's when looking at another scene (enrollment).
        `frame`: an already rendered frame of the same camera (no render here)."""
        if frame is None:
            frame = self.frame(data, renderer, model)
        rgb, depth, points = frame.rgb, frame.depth, frame.points
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        inside = (
            np.isfinite(z) & (x >= self.workspace_x[0]) & (x <= self.workspace_x[1])
            & (y >= self.workspace_y[0]) & (y <= self.workspace_y[1])
        )
        if self.basket_xy is not None:
            inside &= ~((np.abs(x - self.basket_xy[0]) < self.basket_margin) & (np.abs(y - self.basket_xy[1]) < self.basket_margin))
        if frame.robot is not None:
            inside &= ~frame.robot
        if inside.sum() < 1000:
            raise RuntimeError("perception failed: the workspace is not in view")
        heights = z[inside]
        histogram, edges = np.histogram(heights, bins=np.arange(heights.min(), heights.max() + 0.002, 0.002))
        surface = float(edges[int(np.argmax(histogram))] + 0.001)
        mask = inside & (z > surface + ABOVE_SURFACE_M)
        # Split objects that touch in the image: drop pixels on a height step between
        # neighbours (a tuna can 8 mm from a lying can merged into one blob and
        # neither was recognised). The same cut applies at enrollment.
        zz = np.nan_to_num(z, nan=surface)
        step = np.zeros_like(mask)
        step[:, :-1] |= np.abs(np.diff(zz, axis=1)) > HEIGHT_STEP_M
        step[:-1, :] |= np.abs(np.diff(zz, axis=0)) > HEIGHT_STEP_M
        # ... and on a jump in range: where a nearer object occludes a farther one of
        # the same height (an apple in front of an orange) there is no height step at
        # the seam, only a depth one; the two merged into one unidentified blob.
        rng = np.nan_to_num(depth, nan=0.0)
        step[:, :-1] |= np.abs(np.diff(rng, axis=1)) > DEPTH_STEP_M
        step[:-1, :] |= np.abs(np.diff(rng, axis=0)) > DEPTH_STEP_M
        labels, count = ndimage.label(mask & ~step, structure=np.ones((3, 3)))
        touch = ndimage.binary_dilation(frame.robot, iterations=1) if frame.robot is not None else None
        found = []
        for index in range(1, count + 1):
            region = labels == index
            pixels = int(region.sum())
            if pixels < MIN_OBJECT_PIXELS:
                continue
            region_points = points[region]
            found.append(Detection(None, np.inf, region, region_points.mean(axis=0),
                                   self._features(region_points, rgb[region], surface), pixels,
                                   occluded=bool(touch is not None and np.any(region & touch))))
        return found

    @staticmethod
    def _features(points: np.ndarray, colours: np.ndarray, surface: float) -> Features:
        height = float(np.percentile(points[:, 2], 99.0)) - surface
        xy = points[:, :2]
        if len(xy) > 4000:
            xy = xy[np.random.default_rng(0).choice(len(xy), 4000, replace=False)]
        short, long, _ = _min_area_rectangle(xy)
        try:
            hull_area = float(ConvexHull(xy).volume)
        except Exception:
            hull_area = short * long
        fill = hull_area / max(short * long, 1e-9)
        r, g, b = (np.mean(colours[:, :3], axis=0) / 255.0).tolist()
        hue, saturation, _ = colorsys.rgb_to_hsv(r, g, b)
        return Features(height, short, long, fill, hue, saturation)

    def _cost(self, features: Features, references) -> float:
        """Best match over an object's references (one per enrolled pose)."""
        if isinstance(references, Features):
            references = [references]
        return min(self._cost_one(features, reference) for reference in references)

    def _cost_one(self, features: Features, reference: Features) -> float:
        terms = [
            (features.height - reference.height) / SCALES["height"],
            (features.short - reference.short) / SCALES["short"],
            (features.long - reference.long) / SCALES["long"],
            (features.fill - reference.fill) / SCALES["fill"],
            # Hue only means something for coloured surfaces.
            _hue_distance(features.hue, reference.hue) / SCALES["hue"] * min(reference.saturation, 0.5) * 2.0,
        ]
        return float(np.sum(np.square(terms)))

    def detect(self, data: mujoco.MjData, renderer: mujoco.Renderer | None = None, unique: bool = True,
               frame: "CameraFrame | None" = None) -> list[Detection]:
        """Detections with identities; candidates that match nothing well enough keep
        label None. `unique`: each enrolled object at most once (Hungarian); False
        when the table may hold several of the same object (each candidate takes its
        best match)."""
        if not self.references:
            raise RuntimeError("enroll() the known objects first")
        found = self.candidates(data, renderer, frame=frame)
        labels = list(self.references)
        if not found:
            return []
        cost = np.array([[self._cost(d.features, self.references[k]) for k in labels] for d in found])
        if not unique:
            for row, detection in enumerate(found):
                col = int(np.argmin(cost[row]))
                detection.cost = float(cost[row, col])
                if detection.cost <= MAX_COST:
                    detection.label = labels[col]
            return found
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows, cols):
            found[row].cost = float(cost[row, col])
            if cost[row, col] <= MAX_COST:
                found[row].label = labels[col]
        return found

    # ------------------------------------------------------------------ enrollment
    def enroll(self, scene_factory, yaws=(0.0, 45.0)) -> dict[str, list[Features]]:
        """Reference features per known object and pose: `scene_factory(key, yaw, pose)`
        builds a scene with that object alone on the table; its one detection,
        averaged over `yaws`."""
        self.references = {}
        for key, pose in self.known:
            samples = []
            for yaw in yaws:
                scene = scene_factory(key, yaw, pose)
                found = self.candidates(scene.data, model=scene.model)
                if len(found) != 1:
                    raise RuntimeError(f"enrollment of {key}/{pose}: expected one object in view, found {len(found)}")
                samples.append(found[0].features)
            mean = {name: float(np.mean([getattr(s, name) for s in samples])) for name in samples[0].__dict__}
            # Hue is circular: average on the circle.
            angles = [2 * np.pi * s.hue for s in samples]
            mean["hue"] = float((np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))) / (2 * np.pi)) % 1.0)
            self.references.setdefault(key, []).append(Features(**mean))
        return self.references


@dataclass
class Track:
    track_id: int
    label: str
    position: np.ndarray
    history: list[list[float]] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    seen: int = 0
    missed: int = 0

    # Looks in the velocity fit: short enough to follow a belt that has just
    # started (8 looks still averaged in the 1 s ramp at t = 4 s and put the can's
    # arrival 3 s late), long enough to smooth the centroid's few-mm jitter.
    VELOCITY_WINDOW = 4

    @property
    def velocity(self) -> np.ndarray:
        """World velocity (m/s): least-squares slope of the last VELOCITY_WINDOW
        positions against time; zero until two timed looks."""
        if len(self.times) < 2:
            return np.zeros(3)
        t = np.asarray(self.times[-self.VELOCITY_WINDOW:])
        p = np.asarray(self.history[-self.VELOCITY_WINDOW:])
        t = t - t.mean()
        if float(t @ t) < 1e-9:
            return np.zeros(3)
        return (t @ (p - p.mean(axis=0))) / float(t @ t)

    def predict(self, at_time: float) -> np.ndarray:
        if not self.times:
            return self.position.copy()
        return self.position + self.velocity * (at_time - self.times[-1])


class ObjectTracker:
    """Tracking by detection: each detection joins the track with the same label
    whose last position is nearest (within `gate` m), else starts a new track. A
    track not seen for `max_missed` looks in a row is dropped (the object left the
    table -- in the basket, or knocked off)."""

    def __init__(self, gate: float = 0.08, max_missed: int = 1, first_id: int = 1) -> None:
        """`first_id`: where the ids start (two trackers on one camera keep apart)."""
        self.gate, self.max_missed = gate, max_missed
        self.tracks: dict[int, Track] = {}
        self._next = first_id

    def update(self, detections: list[Detection], t: float | None = None) -> list[Track]:
        """`t`: the look's time (s). With it, tracks gate on their predicted position
        (moving objects) and keep a velocity estimate."""
        seen = set()
        for detection in detections:
            if detection.label is None:
                continue
            best, best_distance = None, self.gate
            for track in self.tracks.values():
                if track.label != detection.label or track.track_id in seen:
                    continue
                expected = track.predict(t) if t is not None else track.position
                distance = float(np.linalg.norm(expected[:2] - detection.centroid[:2]))
                if distance <= best_distance:
                    best, best_distance = track, distance
            if best is None:
                best = Track(self._next, detection.label, detection.centroid.copy())
                self.tracks[best.track_id] = best
                self._next += 1
            best.position = detection.centroid.copy()
            best.history.append(np.round(detection.centroid, 4).tolist())
            if t is not None:
                best.times.append(float(t))
            best.seen += 1
            best.missed = 0
            detection.track_id = best.track_id
            seen.add(best.track_id)
        for track_id in list(self.tracks):
            if track_id not in seen:
                self.tracks[track_id].missed += 1
                if self.tracks[track_id].missed > self.max_missed:
                    del self.tracks[track_id]
        return [t for t in self.tracks.values() if t.missed == 0]
