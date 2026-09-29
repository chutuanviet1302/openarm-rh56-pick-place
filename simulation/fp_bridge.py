"""Windows side of the FoundationPose bridge: render what the D435 sees, hand it to
FoundationPose in WSL, read the pose back.

    frame = capture(scene, "can")            # RGB, metric depth, object mask, K, T_world_cam
    frame.export(directory)                  # the files FoundationPose's run_demo reads
    T_world_object = estimate_pose(scene, "can")   # capture + WSL run + frame change

Conventions (the one place they are converted):
    MuJoCo camera frame: looks down -z, +y up. OpenCV / FoundationPose: looks down +z,
    +y down. Same origin, so T_world_cam_cv = T_world_cam_mujoco @ diag(1, -1, -1, 1).
    K from the camera's vertical field of view, square pixels, principal point at the
    image centre (MuJoCo's pinhole).
    FoundationPose returns T_cam_object in the OBJ's own frame = the MuJoCo body frame
    (simulation/objects.py), so T_world_object = T_world_cam_cv @ T_cam_object.

The mask is the simulator's segmentation of the object (the visual mesh the camera
sees) -- a stand-in for a real segmenter (SAM / colour) on the robot; see the Day 3
notes. Depth, RGB, K and the pose all come from the rendered image, not sim state.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from simulation.objects import OBJECTS

WIDTH, HEIGHT = 640, 480
MUJOCO_TO_CV = np.diag([1.0, -1.0, -1.0])
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WSL_DISTRO = os.environ.get("FP_WSL_DISTRO", "Ubuntu-22.04")


@dataclass
class Frame:
    object_key: str
    rgb: np.ndarray          # (H, W, 3) uint8
    depth: np.ndarray        # (H, W) float32, metres along the optical axis, 0 = no return
    mask: np.ndarray         # (H, W) bool, the object's pixels
    K: np.ndarray            # 3x3
    T_world_cam: np.ndarray  # 4x4, OpenCV camera frame

    def export(self, directory: Path) -> Path:
        """Write the frame in the layout wsl/fp_run.py reads."""
        import imageio.v3 as iio

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        iio.imwrite(directory / "rgb.png", self.rgb)
        # FoundationPose's readers take depth as uint16 millimetres; the .npy keeps metres.
        np.save(directory / "depth.npy", self.depth.astype(np.float32))
        iio.imwrite(directory / "depth.png", np.clip(self.depth * 1000.0, 0, 65535).astype(np.uint16))
        iio.imwrite(directory / "mask.png", (self.mask * 255).astype(np.uint8))
        np.savetxt(directory / "cam_K.txt", self.K)
        np.savetxt(directory / "T_world_cam.txt", self.T_world_cam)
        (directory / "meta.json").write_text(json.dumps({
            "object": self.object_key,
            "mesh": str(OBJECTS[self.object_key].mesh_file),
            "mesh_wsl": to_wsl_path(OBJECTS[self.object_key].mesh_file),
            "mask_pixels": int(self.mask.sum()),
        }, indent=2))
        return directory


def intrinsics(model: mujoco.MjModel, camera: str, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    fovy = np.radians(float(model.camera(camera).fovy[0]))
    f = 0.5 * height / np.tan(0.5 * fovy)
    return np.array([[f, 0.0, 0.5 * width], [0.0, f, 0.5 * height], [0.0, 0.0, 1.0]])


def camera_pose_cv(data: mujoco.MjData, model: mujoco.MjModel, camera: str) -> np.ndarray:
    cam = model.camera(camera).id
    transform = np.eye(4)
    transform[:3, :3] = data.cam_xmat[cam].reshape(3, 3) @ MUJOCO_TO_CV
    transform[:3, 3] = data.cam_xpos[cam]
    return transform


def object_geoms(model: mujoco.MjModel, body: int) -> list[int]:
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]


def capture(scene, key: str | None = None, camera: str = "d435_head",
            renderer: mujoco.Renderer | None = None) -> Frame:
    """Render RGB, depth and the object's mask from `camera` at the current state."""
    key = key or scene.pick_object
    model, data = scene.model, scene.data
    own = renderer is None
    renderer = renderer or mujoco.Renderer(model, HEIGHT, WIDTH)
    try:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera)
        depth = renderer.render().astype(np.float32)
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera)
        segmentation = renderer.render()[..., 0].copy()
        renderer.disable_segmentation_rendering()
    finally:
        if own:
            renderer.close()
    body = int(model.joint(scene.object_joints[key]).bodyid[0])
    mask = np.isin(segmentation, object_geoms(model, body))
    key = scene.object_types.get(key, key)  # the registry key: the mesh FoundationPose gets
    far = float(model.vis.map.zfar * model.stat.extent)
    depth[~np.isfinite(depth) | (depth >= 0.99 * far)] = 0.0
    return Frame(key, rgb, depth, mask, intrinsics(model, camera), camera_pose_cv(data, model, camera))


def project(points_world: np.ndarray, frame: Frame) -> np.ndarray:
    """Pixel (u, v) of world points in `frame`'s camera (for checks and overlays)."""
    world_to_cam = np.linalg.inv(frame.T_world_cam)
    cam = points_world @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    uv = cam @ frame.K.T
    return uv[:, :2] / uv[:, 2:3]


def to_wsl_path(path: Path) -> str:
    """D:\\a b\\c -> /mnt/d/a b/c"""
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix()[2:]}"


def run_foundationpose(directory: Path, timeout_s: float = 600.0) -> np.ndarray:
    """Run wsl/fp_run.py on an exported frame; returns T_cam_object (4x4, OpenCV)."""
    # fp_run_dirs.sh sources fp_env.sh: conda env, CUDA paths, libnvrtc, allocator.
    script = to_wsl_path(PROJECT_ROOT / "wsl" / "fp_run_dirs.sh")
    command = ["wsl.exe", "-d", WSL_DISTRO, "--", "bash", script, to_wsl_path(directory)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    pose_file = Path(directory) / "T_cam_object.txt"
    if completed.returncode != 0 or not pose_file.is_file():
        tail = "\n".join((completed.stderr or completed.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"FoundationPose failed (exit {completed.returncode}):\n{tail}")
    return np.loadtxt(pose_file)


def estimate_pose(scene, key: str | None = None, camera: str = "d435_head", workdir: Path | None = None,
                  mask: np.ndarray | None = None) -> np.ndarray:
    """T_world_object from FoundationPose on one rendered frame. `mask`: the object's
    pixels from the detector; without it, the simulator's segmentation."""
    key = key or scene.pick_object
    frame = capture(scene, key, camera)
    if mask is not None:
        frame.mask = np.asarray(mask, dtype=bool)
    if frame.mask.sum() < 100:
        raise RuntimeError(f"perception failed: {key} covers only {int(frame.mask.sum())} px in '{camera}'")
    directory = frame.export(workdir or PROJECT_ROOT / "artifacts" / "fp_frames" / key)
    try:
        return frame.T_world_cam @ run_foundationpose(directory)
    except (RuntimeError, subprocess.TimeoutExpired):
        # The first call after WSL was idle failed once (exit 1) and ran fine straight
        # after: one retry.
        return frame.T_world_cam @ run_foundationpose(directory)
