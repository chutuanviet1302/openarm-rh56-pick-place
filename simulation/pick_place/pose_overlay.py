"""6D pose display: every pose estimate the task makes, drawn as an oriented box and
its x/y/z axes -- on the head camera's image (like Isaac ROS CenterPose / FoundationPose
demos) and, in the replay, as 3-D markers in the MuJoCo viewer.

    listener = PoseEventLog(scene)                  # records every get_object_pose()
    pose_source.POSE_LISTENERS.append(listener)
    ...
    listener.arrays()                               # saved with the replay frames

Colours: box red (the estimate), axes x red, y green, z blue; the ground-truth box is
drawn thin green on the image so the estimate's error is visible at a glance.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

INSET_SIZE = (320, 240)          # width, height of the image kept per estimate
AXIS_LENGTH = 0.06               # m
BOX_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7))
AXIS_COLOURS_BGR = ((0, 0, 255), (0, 200, 0), (255, 0, 0))
AXIS_RGBA = ((1.0, 0.1, 0.1, 1.0), (0.1, 0.9, 0.1, 1.0), (0.1, 0.3, 1.0, 1.0))
BOX_RGBA = (1.0, 0.15, 0.15, 1.0)


def object_box(model: mujoco.MjModel, body: int) -> np.ndarray:
    """(centre xyz, half size xyz) of the body's geoms, in the body frame (mesh
    vertices where the geom is a mesh, the geom's bounding box otherwise)."""
    points = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body:
            continue
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
        rotation = rotation.reshape(3, 3)
        if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[g])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            local = model.mesh_vert[start:start + count]
        else:
            centre, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
            local = centre + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        points.append(local @ rotation.T + model.geom_pos[g])
    points = np.concatenate(points)
    low, high = points.min(axis=0), points.max(axis=0)
    return np.concatenate([(low + high) / 2.0, (high - low) / 2.0])


def box_corners(pose: np.ndarray, box: np.ndarray) -> np.ndarray:
    """8 world corners, index bits (x, y, z) = (4, 2, 1)."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    local = box[:3] + signs * box[3:]
    return local @ pose[:3, :3].T + pose[:3, 3]


def _project(points: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    world_to_cam = np.linalg.inv(T_world_cam)
    cam = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    uv = cam @ K.T
    return uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)


def draw_pose(rgb: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray, pose: np.ndarray, box: np.ndarray,
              truth: np.ndarray | None = None, label: str = "") -> np.ndarray:
    """The camera image with the estimated box + axes (and the ground-truth box, thin)."""
    import cv2

    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if truth is not None:
        uv = _project(box_corners(truth, box), K, T_world_cam).astype(int)
        for a, b in BOX_EDGES:
            cv2.line(image, tuple(uv[a]), tuple(uv[b]), (0, 200, 0), 1, cv2.LINE_AA)
    uv = _project(box_corners(pose, box), K, T_world_cam).astype(int)
    for a, b in BOX_EDGES:
        cv2.line(image, tuple(uv[a]), tuple(uv[b]), (0, 0, 255), 2, cv2.LINE_AA)
    origin = pose[:3, 3] + pose[:3, :3] @ box[:3]
    ends = origin + (pose[:3, :3] * AXIS_LENGTH).T
    uv = _project(np.vstack([origin, ends]), K, T_world_cam).astype(int)
    for axis in range(3):
        cv2.arrowedLine(image, tuple(uv[0]), tuple(uv[axis + 1]), AXIS_COLOURS_BGR[axis], 2, cv2.LINE_AA, tipLength=0.2)
    if label:
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class PoseEventLog:
    """A pose_source listener: for every estimate, the sim time, the object, the pose,
    its box and the annotated head-camera image (also written as PNG)."""

    def __init__(self, scene, png_dir: Path | None = Path("artifacts") / "pose_overlays") -> None:
        self.scene = scene
        self.png_dir = png_dir
        self.times: list[float] = []
        self.names: list[str] = []
        self.poses: list[np.ndarray] = []
        self.boxes: list[np.ndarray] = []
        self.images: list[np.ndarray] = []
        self._renderer: mujoco.Renderer | None = None

    def __call__(self, scene, key: str, backend: str, pose: np.ndarray, camera: str) -> None:
        import cv2

        from simulation.fp_bridge import HEIGHT, WIDTH, camera_pose_cv, intrinsics
        from simulation.pick_place.pose_source import pose_error

        model, data = scene.model, scene.data
        body = int(model.joint(scene.object_joints[key]).bodyid[0])
        box = object_box(model, body)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        self._renderer.update_scene(data, camera=camera)
        rgb = self._renderer.render().copy()
        truth = scene.object_pose(key)
        translation, rotation = pose_error(pose, truth)
        label = f"{backend}: {key}  err {translation * 1000:.0f} mm"
        image = draw_pose(rgb, intrinsics(model, camera), camera_pose_cv(data, model, camera), pose, box, truth, label)
        index = len(self.times)
        self.times.append(float(data.time))
        self.names.append(key)
        self.poses.append(np.asarray(pose, dtype=float).copy())
        self.boxes.append(box)
        self.images.append(cv2.resize(image, INSET_SIZE, interpolation=cv2.INTER_AREA))
        if self.png_dir is not None:
            self.png_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.png_dir / f"{index:02d}_{key}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self.times)
        return dict(
            pose_times=np.asarray(self.times, dtype=float),
            pose_names=np.asarray(self.names, dtype=str),
            pose_mats=np.asarray(self.poses).reshape(n, 4, 4),
            pose_boxes=np.asarray(self.boxes).reshape(n, 6),
            pose_images=np.asarray(self.images, dtype=np.uint8).reshape(n, INSET_SIZE[1], INSET_SIZE[0], 3),
        )


def add_pose_markers(scn: mujoco.MjvScene, pose: np.ndarray, box: np.ndarray) -> None:
    """Oriented box edges and x/y/z arrows at `pose` into the viewer's user scene."""
    def connector(kind, width, a, b, rgba) -> None:
        if scn.ngeom >= scn.maxgeom:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, kind, width, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
        scn.ngeom += 1

    corners = box_corners(pose, box)
    for a, b in BOX_EDGES:
        connector(mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, corners[a], corners[b], BOX_RGBA)
    origin = pose[:3, 3] + pose[:3, :3] @ box[:3]
    for axis in range(3):
        connector(mujoco.mjtGeom.mjGEOM_ARROW, 0.004, origin, origin + pose[:3, axis] * AXIS_LENGTH * 1.5, AXIS_RGBA[axis])
