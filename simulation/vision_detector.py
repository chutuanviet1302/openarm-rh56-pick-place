from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from simulation.five_finger_model import OBJECT_HALF_HEIGHT, OBJECT_RADIUS, TABLE_TOP_Z


@dataclass
class ObjectDetectionResult:
    found: bool
    pos_world: np.ndarray
    radius_m: float
    height_m: float
    confidence: float
    pixel_count: int = 0
    pixel_center: tuple[int, int] | None = None
    annotated_image: np.ndarray | None = None


def red_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels where red clearly dominates: the soup can's label against a wood table."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    return (r > 100) & (r > 1.4 * g) & (r > 1.4 * b)


def fit_circle_known_radius(points_xy: np.ndarray, radius: float, initial: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Least-squares centre of a circle of known radius through surface points.

    The camera only ever sees one side of the can, so the centroid of the visible
    points lies on the near surface, not at the axis. Minimising sum((|p - c| - r)^2)
    with Gauss-Newton pulls the centre back to where a cylinder of the known radius
    actually has to be to show that surface.
    """
    centre = np.asarray(initial, dtype=float).copy()
    for _ in range(iterations):
        delta = points_xy - centre
        distance = np.linalg.norm(delta, axis=1)
        distance[distance == 0] = 1e-9
        residual = distance - radius
        jacobian = -delta / distance[:, None]
        step = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        centre += step
        if np.linalg.norm(step) < 1e-6:
            break
    return centre


class VisionDetector:
    """RGB-D detection of the pick object from a MuJoCo camera (sim stand-in for the D435).

    Same pipeline shape as `openarm_pick_place.perception.estimate_object_pose` on the
    real robot: colour segmentation -> aligned depth -> pinhole deprojection -> camera to
    world transform. Nothing here reads the object's simulated pose; the position comes
    from pixels and depth only, so a wrong camera calibration or a bad mask shows up as
    a measurable error instead of being papered over.
    """

    def __init__(self, model: mujoco.MjModel, camera_name: str = "d435_head", width: int = 640, height: int = 480) -> None:
        self.model = model
        self.camera_name = camera_name
        self.cam_id = model.camera(camera_name).id
        self.width = width
        self.height = height
        # One renderer, toggled between colour and depth: a second Renderer instance
        # per model returned far-plane depth for every pixel once more than one model
        # had been rendered in the process.
        self.renderer = mujoco.Renderer(model, height, width)
        # Pinhole intrinsics from the camera's vertical field of view (MuJoCo has no
        # separate horizontal fov; square pixels are assumed, as on the D435).
        fovy = float(model.cam_fovy[self.cam_id])
        self.fy = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
        self.fx = self.fy
        self.cx, self.cy = 0.5 * width, 0.5 * height

    def deproject(self, data: mujoco.MjData, rows: np.ndarray, columns: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Pixels + metric depth -> world points. MuJoCo cameras look down their own -z
        with +y up, so image rows (downward) map to -y."""
        x = (columns - self.cx) * depth / self.fx
        y = -(rows - self.cy) * depth / self.fy
        z = -depth
        camera_points = np.column_stack((x, y, z))
        rotation = data.cam_xmat[self.cam_id].reshape(3, 3)
        return camera_points @ rotation.T + data.cam_xpos[self.cam_id]

    def detect_object(self, data: mujoco.MjData, render_annotation: bool = True) -> ObjectDetectionResult:
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(data, camera=self.camera_name)
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()

        mask = red_mask(rgb) & np.isfinite(depth) & (depth > 0)
        rows, columns = np.nonzero(mask)
        if rows.size < 50:
            return ObjectDetectionResult(False, np.full(3, np.nan), OBJECT_RADIUS, 2 * OBJECT_HALF_HEIGHT, 0.0, int(rows.size))

        values = depth[rows, columns]
        # Reject depth outliers (mask bleeding onto the table edge or the basket).
        median = np.median(values)
        deviation = np.median(np.abs(values - median))
        keep = np.ones(values.shape, dtype=bool) if deviation == 0 else np.abs(values - median) <= 3 * deviation
        rows, columns, values = rows[keep], columns[keep], values[keep]

        world = self.deproject(data, rows.astype(float), columns.astype(float), values.astype(float))
        # Tabletop assumption: the object stands on the known table plane, so its centre
        # height follows from its height; only x/y have to come from the image.
        centre_z = TABLE_TOP_Z + OBJECT_HALF_HEIGHT
        surface_xy = world[:, :2]
        camera_xy = data.cam_xpos[self.cam_id][:2]
        centroid = surface_xy.mean(axis=0)
        away = centroid - camera_xy
        away /= max(np.linalg.norm(away), 1e-9)
        centre_xy = fit_circle_known_radius(surface_xy, OBJECT_RADIUS, centroid + 0.5 * OBJECT_RADIUS * away)
        pos_world = np.array([centre_xy[0], centre_xy[1], centre_z])

        u_center, v_center = int(np.mean(columns)), int(np.mean(rows))
        annotated = None
        if render_annotation:
            radius_px = int(0.5 * (columns.max() - columns.min()))
            pil_img = Image.fromarray(rgb.copy())
            draw = ImageDraw.Draw(pil_img)
            bbox = [u_center - radius_px, v_center - radius_px, u_center + radius_px, v_center + radius_px]
            draw.ellipse(bbox, outline=(0, 255, 0), width=3)
            draw.line([(u_center - 10, v_center), (u_center + 10, v_center)], fill=(0, 255, 255), width=2)
            draw.line([(u_center, v_center - 10), (u_center, v_center + 10)], fill=(0, 255, 255), width=2)
            text = f"Obj: [{pos_world[0]:.3f}, {pos_world[1]:.3f}, {pos_world[2]:.3f}]m  n={rows.size}"
            draw.text((max(0, u_center - 60), max(0, v_center - radius_px - 20)), text, fill=(255, 255, 0))
            annotated = np.array(pil_img)

        return ObjectDetectionResult(
            found=True,
            pos_world=pos_world,
            radius_m=OBJECT_RADIUS,
            height_m=2 * OBJECT_HALF_HEIGHT,
            confidence=float(min(1.0, rows.size / 2000.0)),
            pixel_count=int(rows.size),
            pixel_center=(u_center, v_center),
            annotated_image=annotated,
        )
