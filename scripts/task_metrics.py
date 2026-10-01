"""Measure a recorded task run: timing, joint-limit margin, smoothness, contacts.

Works on any FrameRecorder recording (artifacts/*_frames.npz), old ones included: the
contacts are recomputed offline from the recorded qpos (mj_fwdPosition runs collision
detection on the recorded configuration), so nothing has to be logged during the run.

    python -m scripts.task_metrics artifacts/bin_conveyor_frames.npz
    python -m scripts.task_metrics artifacts/bin_task_frames.npz --task bin_task
    python -m scripts.task_metrics FRAMES --report artifacts/bin_conveyor.json --out metrics.json

What is measured (the definitions are the metric, not tuned values):
    timing      sim seconds per caption segment, total; wall seconds from the report
    limits      per arm joint, the smallest distance to its range end over the run (deg)
    smoothness  arm joint velocity / acceleration / jerk by finite differences of the
                recorded positions (np.gradient over the recorded times); RMS and max.
                "stops": how often an arm goes from moving (speed norm > MOVING_RAD_S)
                to standing still (< STILL_RAD_S) -- each is a stop-and-go in the motion.
    contacts    robot geoms (arm links + hands) penetrating (dist < 0) the table / work
                platform / belt, the box, the robot's own pedestal or torso, the other
                arm, and any object other than the current target (the target is the
                name in the caption, "... -> kind (name)"); arm links (not the hand)
                touching any object at all. Frames and deepest penetration (mm) each.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import mujoco
import numpy as np

MOVING_RAD_S = 0.05
STILL_RAD_S = 0.01
TASKS = {"bin_conveyor": "simulation.pick_place.bin_conveyor_task", "bin_task": "simulation.pick_place.bin_task"}
TARGET_IN_CAPTION = re.compile(r"->\s*\S+\s*\((\w+)\)")


def _scene_for(recording, task: str):
    module = importlib.import_module(TASKS.get(task, task))
    if "layout" in recording.files and hasattr(module, "scene_from_layout"):
        return module.scene_from_layout(json.loads(str(recording["layout"])))
    return module.build_scene()


def segment_times(times: np.ndarray, captions: np.ndarray) -> list[dict]:
    segments, start = [], 0
    for index in range(1, len(captions) + 1):
        if index == len(captions) or captions[index] != captions[start]:
            segments.append({"caption": str(captions[start]), "start_s": round(float(times[start]), 2),
                             "seconds": round(float(times[min(index, len(times) - 1)] - times[start]), 2)})
            start = index
    return segments


def phase_totals(times: np.ndarray, phases: np.ndarray) -> dict:
    """Sim seconds spent in each demo phase ("<arm> <phase>"), summed over the run."""
    totals: dict[str, float] = {}
    dt = np.diff(times, append=times[-1])
    for phase, seconds in zip(phases, dt):
        key = str(phase) or "(none)"
        totals[key] = totals.get(key, 0.0) + float(seconds)
    return {k: round(v, 1) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}


def joint_metrics(model: mujoco.MjModel, scene, times: np.ndarray, qpos: np.ndarray) -> dict:
    out = {}
    for side in ("right", "left"):
        addresses = np.asarray(scene.arm_qpos[side], dtype=int)
        joints = [int(np.flatnonzero(model.jnt_qposadr == a)[0]) for a in addresses]
        q = qpos[:, addresses]
        low, high = model.jnt_range[joints, 0], model.jnt_range[joints, 1]
        margin = np.minimum(q - low, high - q)                      # (frames, 7)
        worst = int(np.argmin(margin.min(axis=0)))
        velocity = np.gradient(q, times, axis=0)
        acceleration = np.gradient(velocity, times, axis=0)
        jerk = np.gradient(acceleration, times, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        stops, moving = 0, False
        for s in speed:
            if not moving and s > MOVING_RAD_S:
                moving = True
            elif moving and s < STILL_RAD_S:
                moving = False
                stops += 1
        out[side] = {
            "min_joint_margin_deg": round(float(np.degrees(margin.min())), 2),
            "min_margin_joint": model.joint(joints[worst]).name,
            "margin_per_joint_deg": np.round(np.degrees(margin.min(axis=0)), 2).tolist(),
            "max_speed_rad_s": round(float(np.abs(velocity).max()), 3),
            "rms_accel_rad_s2": round(float(np.sqrt(np.mean(acceleration ** 2))), 3),
            "max_accel_rad_s2": round(float(np.abs(acceleration).max()), 2),
            "rms_jerk_rad_s3": round(float(np.sqrt(np.mean(jerk ** 2))), 2),
            "p99_jerk_rad_s3": round(float(np.percentile(np.abs(jerk), 99)), 2),
            "stops": stops,
        }
    return out


def contact_metrics(model: mujoco.MjModel, scene, qpos: np.ndarray, captions: np.ndarray) -> dict:
    data = mujoco.MjData(model)
    side = np.array([{"left": 0, "right": 1}.get(scene.robot_side(g), -1) for g in range(model.ngeom)])
    robot = np.array([scene._is_robot_geom(g) for g in range(model.ngeom)])
    hand = np.array([scene.hand_side(g) is not None for g in range(model.ngeom)])
    object_of = {scene.object_geom_of(name): name for name in scene.object_joints}
    category = {}
    for g in scene.table_geoms:
        category[g] = "table_platform_belt"
    for g in scene.basket_geoms:
        category[g] = "box"
    for g in scene.robot_body_geoms:
        category[g] = "robot_body"
    mounts = {model.body(f"openarm_{s}_link{i}").id for s in ("left", "right") for i in (0, 1)}
    kinds = ("table_platform_belt", "box", "robot_body", "inter_arm", "non_target_object", "arm_link_object")
    frames = {k: 0 for k in kinds}
    deepest = {k: 0.0 for k in kinds}
    worst_at = {k: None for k in kinds}
    target = None
    for index, (q, caption) in enumerate(zip(qpos, captions)):
        found = TARGET_IN_CAPTION.search(str(caption))
        if found:
            target = found.group(1)
        data.qpos[:] = q
        mujoco.mj_fwdPosition(model, data)
        hit = set()
        for contact in data.contact[: data.ncon]:
            g1, g2, dist = int(contact.geom1), int(contact.geom2), float(contact.dist)
            if dist >= 0.0:
                continue
            for a, b in ((g1, g2), (g2, g1)):
                if not robot[a]:
                    continue
                kind = None
                if robot[b]:
                    if side[a] >= 0 and side[b] >= 0 and side[a] != side[b]:
                        kind = "inter_arm"
                elif b in category:
                    kind = category[b]
                    if kind == "robot_body" and int(model.geom_bodyid[a]) in mounts:
                        kind = None
                elif b in object_of:
                    if not hand[a]:
                        kind = "arm_link_object"
                    elif object_of[b] != target:
                        kind = "non_target_object"
                if kind is None:
                    continue
                hit.add(kind)
                if -dist * 1000.0 > deepest[kind]:
                    deepest[kind] = -dist * 1000.0
                    worst_at[kind] = {"frame": index, "robot": model.body(int(model.geom_bodyid[a])).name,
                                      "other": model.geom(b).name or model.body(int(model.geom_bodyid[b])).name}
        for kind in hit:
            frames[kind] += 1
    return {k: {"frames": frames[k], "deepest_mm": round(deepest[k], 2), "deepest_at": worst_at[k]} for k in kinds}


def measure(frames_path: Path, task: str, report_path: Path | None) -> dict:
    recording = np.load(frames_path)
    times, qpos, captions = recording["times"], recording["qpos"], recording["captions"]
    scene = _scene_for(recording, task)
    model = scene.model
    result = {
        "frames_file": str(frames_path),
        "frames": int(len(times)),
        "sim_seconds": round(float(times[-1] - times[0]), 2),
        "segments": segment_times(times, captions),
        "phases": phase_totals(times, recording["phases"]) if "phases" in recording.files else None,
        "joints": joint_metrics(model, scene, times, qpos),
        "contacts": contact_metrics(model, scene, qpos, captions),
    }
    if report_path is not None and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result["report"] = {
            "file": str(report_path),
            "in_box": f"{sum(report['in_box'].values())}/{len(report['in_box'])}",
            "sim_seconds": round(report["sim_seconds"], 1),
            "wall_seconds": round(report["wall_seconds"], 1),
            "pose_backend": report.get("pose_backend"),
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Timing, joint margin, smoothness and contacts of a recorded run")
    parser.add_argument("frames", type=Path)
    parser.add_argument("--task", default="bin_conveyor", help="bin_conveyor, bin_task or a module path")
    parser.add_argument("--report", type=Path, help="the run's report JSON (default: derived from the frames name)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = args.report or args.frames.with_name(args.frames.name.replace("_frames.npz", ".json"))
    result = measure(args.frames, args.task, report)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    summary = {k: v for k, v in result.items() if k != "segments"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
