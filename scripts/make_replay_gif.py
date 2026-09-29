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

from simulation.pick_place.bin_conveyor_task import build_scene


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path)
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--size", type=int, nargs=2, default=(640, 400))
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "bin_conveyor.gif")
    args = parser.parse_args(argv)
    rec = np.load(args.frames)
    times, qpos, captions = rec["times"], rec["qpos"], rec["captions"]
    scene = build_scene()
    model, data = scene.model, scene.data
    width, height = args.size
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.30, 0.0, 0.20]
    cam.azimuth, cam.elevation, cam.distance = 180.0, -35.0, 1.45
    step = args.speed / args.fps  # sim seconds per GIF frame
    frames, t_next = [], float(times[0])
    for t, q, text in zip(times, qpos, captions):
        if t < t_next:
            continue
        t_next += step
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        renderer.update_scene(data, camera=cam)
        img = renderer.render().copy()
        label = f"t={t:5.1f}s  x{args.speed:g}  {text}"
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(img)
    imageio.mimsave(args.out, frames, duration=1.0 / args.fps, loop=0)
    print(f"{len(frames)} frames -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
