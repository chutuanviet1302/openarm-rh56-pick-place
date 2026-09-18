"""Record one pick-and-place episode for the web viewer (viewer/index.html).

Runs the demo headless, films it from every scene camera at a fixed frame rate and
writes, under artifacts/episodes/<name>/:

    <camera>.mp4     one clip per camera, all cut from the same sim clock
    perception.png   what the head camera saw at PERCEIVE, with the detection drawn
    episode.json     layout, result, the phase/note timeline and per-frame telemetry
                     (object pose, wrist position, finger forces)

and adds the episode to artifacts/episodes/index.json, which the viewer lists.

    python -m scripts.record_episode --perception
    python -m scripts.record_episode --object 0.22 0.0 --basket 0.24 -0.26 --name centre
    python -m scripts.serve_viewer          # then open http://localhost:8000/viewer/
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from simulation.five_finger_model import BASKET_POSITION_B, PICK_POSITION_A
from simulation.pick_place.demo import Demo, run_trial
from simulation.pick_place.kinematics import upright_tilt_degrees
from simulation.vision_detector import VisionDetector

EPISODES_ROOT = Path("artifacts") / "episodes"
DEFAULT_CAMERAS = ("isometric", "d435_head", "overhead", "close_grasp", "right_wrist_camera")
EXTRA_CAMERAS = ("front_view", "side_view")
WIDTH, HEIGHT = 1280, 960  # drawing is vertex-bound here, so the larger frame is nearly free


class Recorder:
    """Executor step hook: renders every camera once per video frame of sim time and
    samples the telemetry alongside."""

    def __init__(self, demo: Demo, cameras: tuple[str, ...], fps: float, out_dir: Path) -> None:
        self.demo = demo
        self.cameras = cameras
        self.frame_seconds = 1.0 / fps
        # Offscreen drawing here lands on the laptop's integrated GPU, where the arm's
        # 840k-vertex meshes take ~140ms a frame without shadows and ~650ms with the
        # vendor scene's 8192-texel shadow map: shadows, reflections and multisampling
        # are turned off for the recording, and the default rate is 10 fps.
        demo.model.vis.quality.offsamples = 0
        demo.model.vis.global_.offwidth = max(int(demo.model.vis.global_.offwidth), WIDTH)
        demo.model.vis.global_.offheight = max(int(demo.model.vis.global_.offheight), HEIGHT)
        self.renderer = mujoco.Renderer(demo.model, HEIGHT, WIDTH)
        self.writers = {
            camera: imageio.get_writer(
                str(out_dir / f"{camera}.mp4"), fps=fps, codec="libx264", quality=9, pixelformat="yuv420p", macro_block_size=1
            )
            for camera in cameras
        }
        self.next_frame_time = 0.0
        self.frames = 0
        self.telemetry: dict[str, list] = {
            "t": [], "phase": [], "object": [], "object_tilt_deg": [], "wrist": [], "forces": []
        }

    def on_step(self, executor) -> None:
        now = float(self.demo.data.time)
        if now + 1e-9 < self.next_frame_time:
            return
        self.next_frame_time += self.frame_seconds
        self.capture(now)

    def capture(self, now: float) -> None:
        scene, data = self.demo.scene, self.demo.data
        for camera in self.cameras:
            self.renderer.update_scene(data, camera=camera)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0  # update_scene resets the flags
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            self.writers[camera].append_data(self.renderer.render())
        forces = scene.finger_contact_forces("right")
        self.telemetry["t"].append(round(now, 3))
        self.telemetry["phase"].append(self.demo.log.current_phase)
        self.telemetry["object"].append(np.round(scene.object_position(), 4).tolist())
        self.telemetry["object_tilt_deg"].append(round(upright_tilt_degrees(scene.object_quaternion()), 1))
        self.telemetry["wrist"].append(np.round(scene.wrist_position("right"), 4).tolist())
        self.telemetry["forces"].append([round(float(forces[name]), 2) for name in ("thumb", "index", "middle", "ring", "pinky")])
        self.frames += 1

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.renderer.close()


def save_perception_image(demo: Demo, path: Path) -> bool:
    """Re-run the head-camera detection with the annotation drawn, from the pose the
    episode perceived from (attention stance, before the arm moves)."""
    if not demo.perception:
        return False
    result = VisionDetector(demo.model, demo.perception_camera).detect_object(demo.data, render_annotation=True)
    if result.annotated_image is None:
        return False
    imageio.imwrite(str(path), result.annotated_image)
    return True


def record(args: argparse.Namespace) -> Path:
    name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = EPISODES_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    demo = Demo(pick_position=tuple(args.object), basket_position=tuple(args.basket), perception=args.perception)
    demo.scene.reset()
    perception_image = save_perception_image(demo, out_dir / "perception.png")

    cameras = tuple(camera for camera in args.cameras if _has_camera(demo.model, camera))
    recorder = Recorder(demo, cameras, args.fps, out_dir)
    demo.executor.on_step = recorder.on_step
    started = time.perf_counter()
    result = run_trial(demo)
    recorder.capture(float(demo.data.time))  # the final resting frame
    recorder.close()
    wall = time.perf_counter() - started

    values = demo.log.values
    episode = {
        "name": name,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "layout": {"object": list(args.object), "basket": list(args.basket), "perception": args.perception},
        "fps": args.fps,
        "frames": recorder.frames,
        "duration_s": round(float(demo.data.time), 3),
        "wall_seconds": round(wall, 1),
        "cameras": list(cameras),
        "perception_image": "perception.png" if perception_image else None,
        "result": {
            "passed": result.success,
            "failure": result.failure_reason,
            "failed_phase": result.failed_phase,
            "placement_error_mm": round(result.placement_error_m * 1000, 1),
            "final_tilt_deg": round(result.bottle_tilt_deg, 1),
            "perception_error_mm": round(values["perception_error_m"] * 1000, 1) if "perception_error_m" in values else None,
            "grasp_yaw_deg": values.get("grasp_yaw_deg"),
            "place_yaw_deg": values.get("place_yaw_deg"),
            "grasp_forces_N": values.get("grasp_forces"),
            "proof_lift_tilt_deg": values.get("proof_lift_tilt_deg"),
        },
        "events": demo.log.events,
        "telemetry": recorder.telemetry,
    }
    (out_dir / "episode.json").write_text(json.dumps(episode, indent=1), encoding="utf-8")
    _update_index(name, episode)
    return out_dir


def _has_camera(model: mujoco.MjModel, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0


def _update_index(name: str, episode: dict) -> None:
    index_path = EPISODES_ROOT / "index.json"
    entries = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else []
    entries = [entry for entry in entries if entry["name"] != name]
    entries.append({
        "name": name,
        "recorded_at": episode["recorded_at"],
        "layout": episode["layout"],
        "passed": episode["result"]["passed"],
        "placement_error_mm": episode["result"]["placement_error_mm"],
        "duration_s": episode["duration_s"],
    })
    entries.sort(key=lambda entry: entry["recorded_at"], reverse=True)
    index_path.write_text(json.dumps(entries, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", type=float, nargs=2, metavar=("X", "Y"), default=list(PICK_POSITION_A))
    parser.add_argument("--basket", type=float, nargs=2, metavar=("X", "Y"), default=list(BASKET_POSITION_B))
    parser.add_argument("--perception", action="store_true", help="object position from the head camera (RGB-D)")
    parser.add_argument("--name", help="episode folder name (default: timestamp)")
    parser.add_argument("--fps", type=float, default=10.0, help="video frames per second of sim time (default 10)")
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS),
                        help=f"scene cameras to film (default {' '.join(DEFAULT_CAMERAS)}; also {' '.join(EXTRA_CAMERAS)})")
    args = parser.parse_args()
    out_dir = record(args)
    print(f"\nrecorded -> {out_dir}\nview it:   python -m scripts.serve_viewer")


if __name__ == "__main__":
    main()
