"""Play a FrameRecorder recording in the MuJoCo viewer, smoothly.

The recording holds the scene's qpos every ~1/60 s of sim time. Playback is driven by
the wall clock, not by the frame list: every display tick the sim time to show is
(wall time since start) x speed, and the pose is interpolated between the two recorded
frames around it (linear for hinge/slide joints, slerp for free/ball joint
quaternions). A slow machine therefore skips poses instead of slowing down, and a fast
one draws in-between poses instead of repeating frames.

    play(path, scene_builder)                       # real speed; scene_builder(recording) -> Scene
    play(path, scene_builder, speed=2.0, quality="fast")

Overlays (6D pose estimates, detections) are objects with `markers(scn, t)` (add
geoms to the viewer's user scene) and `image(t)` -> (kind, index, RGB array) or None;
their images are shown side by side along the bottom edge, re-uploaded only when one
changes.

At the end the measured update rate is printed and written next to the recording
(<name>_playback.json): ticks per second, and the share of ticks later than 1.5x the
display period -- a number for "smooth", not an impression.
"""

from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

FALLBACK_REFRESH_HZ = 60.0


def display_refresh_hz() -> float:
    """The primary display's refresh rate (Windows GetDeviceCaps VREFRESH); 60 elsewhere."""
    try:
        user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
        dc = user32.GetDC(0)
        hz = gdi32.GetDeviceCaps(dc, 116)  # VREFRESH
        user32.ReleaseDC(0, dc)
        return float(hz) if hz > 1 else FALLBACK_REFRESH_HZ
    except (AttributeError, OSError):
        return FALLBACK_REFRESH_HZ


def gl_renderer() -> str:
    try:
        from OpenGL import GL

        context = mujoco.GLContext(64, 64)
        context.make_current()
        return GL.glGetString(GL.GL_RENDERER).decode()
    except Exception:  # noqa: BLE001 - informational only
        return "unknown (Windows: Settings > System > Display > Graphics > python.exe > High performance)"


def quaternion_slots(model: mujoco.MjModel) -> list[int]:
    """qpos addresses where a unit quaternion starts (free and ball joints)."""
    slots = []
    for joint in range(model.njnt):
        kind, address = int(model.jnt_type[joint]), int(model.jnt_qposadr[joint])
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            slots.append(address + 3)
        elif kind == mujoco.mjtJoint.mjJNT_BALL:
            slots.append(address)
    return slots


def _slerp(a: np.ndarray, b: np.ndarray, f: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + f * (b - a)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1.0 - f) * theta) * a + np.sin(f * theta) * b) / np.sin(theta)


def interpolate(qpos: np.ndarray, times: np.ndarray, t: float, quats: list[int]) -> tuple[np.ndarray, int]:
    """The pose at sim time `t` and the index of the recorded frame at or before it."""
    i = int(np.searchsorted(times, t, side="right")) - 1
    if i < 0:
        return qpos[0], 0
    if i >= len(times) - 1:
        return qpos[-1], len(times) - 1
    span = float(times[i + 1] - times[i])
    f = 0.0 if span <= 0.0 else (t - float(times[i])) / span
    q = qpos[i] + f * (qpos[i + 1] - qpos[i])
    for s in quats:
        q[s:s + 4] = _slerp(qpos[i, s:s + 4], qpos[i + 1, s:s + 4], f)
    return q, i


def _show_images(viewer, images: list[np.ndarray]) -> None:
    try:
        if not images:
            viewer.clear_images()
            return
        x, placed = 0, []
        for image in images:
            placed.append((mujoco.MjrRect(x, 0, image.shape[1], image.shape[0]), np.ascontiguousarray(image)))
            x += image.shape[1]
        viewer.set_images(placed)
    except (AttributeError, TypeError) as error:
        print(f"(camera inset unavailable: {error})")


def apply_quality(viewer, quality: str) -> None:
    """'fast': no shadows or reflections (the most expensive passes on a laptop GPU)."""
    if quality == "fast":
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0


def play(path: Path, scene_builder, speed: float = 1.0, quality: str = "high", show_ui: bool = False,
         overlays: tuple[Callable, ...] = (), camera: dict | None = None, hold: bool = True) -> dict:
    import mujoco.viewer

    try:
        ctypes.windll.winmm.timeBeginPeriod(1)  # 1 ms sleeps (Windows default is 15.6 ms)
    except (AttributeError, OSError):
        pass
    recording = np.load(path)
    times, qpos, captions = recording["times"], recording["qpos"].copy(), recording["captions"]
    scene = scene_builder(recording)
    model, data = scene.model, scene.data
    quats = quaternion_slots(model)
    period = 1.0 / display_refresh_hz()
    ticks: list[float] = []
    stats: dict = {}
    with mujoco.viewer.launch_passive(model, data, show_left_ui=show_ui, show_right_ui=show_ui) as viewer:
        renderer = gl_renderer()
        print("OpenGL renderer:", renderer)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        view = camera or {"lookat": [0.30, 0.0, 0.20], "azimuth": 180.0, "elevation": -35.0, "distance": 1.45}
        viewer.cam.lookat[:] = view["lookat"]
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = view["azimuth"], view["elevation"], view["distance"]
        with viewer.lock():
            apply_quality(viewer, quality)
        start_wall, start_sim, end_sim = time.perf_counter(), float(times[0]), float(times[-1])
        caption, shown = None, None
        while viewer.is_running():
            tick = time.perf_counter()
            t = start_sim + (tick - start_wall) * speed
            if t > end_sim:
                break
            q, index = interpolate(qpos, times, t, quats)
            with viewer.lock():
                data.qpos[:] = q
                data.time = t
                mujoco.mj_kinematics(model, data)
                viewer.user_scn.ngeom = 0
                for overlay in overlays:
                    overlay.markers(viewer.user_scn, t)
                images = [image for image in (overlay.image(t) for overlay in overlays) if image is not None]
                keys = tuple(image[:2] for image in images)
                if keys != shown:
                    shown = keys
                    _show_images(viewer, [image[2] for image in images])
                text = str(captions[index])
                if text != caption:
                    caption = text
                    try:
                        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                          "", text))
                    except (AttributeError, TypeError):
                        pass
            viewer.sync()
            ticks.append(tick)
            spare = period - (time.perf_counter() - tick)
            if spare > 0.0:
                time.sleep(spare)
        if len(ticks) > 2:
            gaps = np.diff(ticks)
            stats = {
                "renderer": renderer,
                "quality": quality,
                "speed": speed,
                "display_hz": round(1.0 / period, 1),
                "ticks_per_s": round(float(len(gaps) / (ticks[-1] - ticks[0])), 1),
                "late_ticks_pct": round(float(np.mean(gaps > 1.5 * period) * 100.0), 2),
                "p99_gap_ms": round(float(np.percentile(gaps, 99) * 1000.0), 1),
                "played_sim_s": round(min(end_sim, start_sim + (ticks[-1] - start_wall) * speed) - start_sim, 1),
            }
            print("playback:", json.dumps(stats))
            out = Path(path).with_name(Path(path).stem.replace("_frames", "") + "_playback.json")
            out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        if hold:
            print("playback finished; close the viewer window to exit")
        while hold and viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    return stats

