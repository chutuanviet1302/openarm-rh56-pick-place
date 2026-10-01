"""Render a saved task recording (FrameRecorder .npz) to a sped-up GIF.

    python -m scripts.make_replay_gif artifacts/bin_conveyor_mink_frames.npz --speed 10 --out artifacts/bin_conveyor_x10.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np



def render_gif(frames_path: Path, out: Path, speed: float = 10.0, fps: int = 15, size=(640, 400),
               detections: bool = True) -> Path:
    """The recording as a sped-up GIF (scene from the recording's layout if it has one),
    with the detection boxes of the latest look drawn in (perception_loop)."""
    from simulation.pick_place.bin_conveyor_task import _builder_for
    from simulation.pick_place.perception_loop import DetectionReplayOverlay

    rec = np.load(frames_path)
    times, qpos, captions = rec["times"], rec["qpos"], rec["captions"]
    scene = _builder_for(frames_path)()
    model, data = scene.model, scene.data
    overlay = DetectionReplayOverlay(rec) if detections else None
    width, height = size
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.30, 0.0, 0.20]
    cam.azimuth, cam.elevation, cam.distance = 180.0, -35.0, 1.45
    step = speed / fps  # sim seconds per GIF frame
    frames, t_next = [], float(times[0])
    for t, q, text in zip(times, qpos, captions):
        if t < t_next:
            continue
        t_next += step
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        renderer.update_scene(data, camera=cam)
        if overlay is not None:
            overlay.markers(renderer.scene, float(t))
        img = renderer.render().copy()
        label = f"t={t:5.1f}s  x{speed:g}  {text}"
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(img)
    renderer.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, duration=1.0 / fps, loop=0)
    print(f"{len(frames)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path)
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--size", type=int, nargs=2, default=(640, 400))
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "bin_conveyor.gif")
    parser.add_argument("--no-detections", action="store_true")
    args = parser.parse_args(argv)
    render_gif(args.frames, args.out, args.speed, args.fps, tuple(args.size), detections=not args.no_detections)


if __name__ == "__main__":
    main()
