"""FoundationPose accuracy against ground truth on rendered D435 frames.

Random scenes on the centre-basket layout: each object in each rest pose at a random
yaw and a random spot in a disc around the pick point. Every frame is exported
(simulation/fp_bridge.py), FoundationPose runs once over all of them in WSL (models
load once), and each estimate is scored against the simulator pose, symmetry-aware:
can = its axis only, fruit = position only, pear = full rotation.

    python -m scripts.eval_pose_accuracy                  # 20 scenes per object/pose
    python -m scripts.eval_pose_accuracy --scenes 5 --objects can
    python -m scripts.eval_pose_accuracy --export-only    # frames only (run FP elsewhere)
    python -m scripts.eval_pose_accuracy --score-only     # score frames FP already ran on

Gate (plan, Day 3): translation < 10 mm and axis/rotation < 10 deg.
Writes artifacts/benchmarks/pose_accuracy.json and frames under artifacts/fp_eval/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from simulation.fp_bridge import PROJECT_ROOT, WSL_DISTRO, capture, to_wsl_path
from simulation.objects import OBJECTS
from simulation.pick_place.grasp_library import select_grasp
from simulation.pick_place.pose_source import pose_error
from simulation.pick_place.scene import Scene

PLATFORM = 0.10
BASKET = (0.28, 0.0)
PICK = (0.28, -0.25)
JITTER_M = 0.05
# The task object set (config/grasp_library.yaml): YCB 005, 007, 009, 017.
CASES = [("can", "upright"), ("tuna_can", "upright"), ("gelatin_box", "upright"), ("orange", "upright")]
FRAMES = PROJECT_ROOT / "artifacts" / "fp_eval"
TRANSLATION_GATE_MM = 10.0
ROTATION_GATE_DEG = 10.0


def export_frames(scenes: int, objects: list[str] | None, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    items = []
    for key, pose in CASES:
        if objects and key not in objects:
            continue
        for index in range(scenes):
            angle, radius = rng.uniform(0, 2 * np.pi), JITTER_M * np.sqrt(rng.uniform())
            xy = (PICK[0] + radius * np.cos(angle), PICK[1] + radius * np.sin(angle))
            yaw = float(rng.uniform(0.0, 360.0))
            scene = Scene(xy, BASKET, work_platform_height=PLATFORM, pick_object=key, pick_pose=pose, pick_yaw_deg=yaw)
            frame = capture(scene, key)
            directory = frame.export(FRAMES / f"{key}_{pose}_{index:02d}")
            np.savetxt(directory / "T_world_object_gt.txt", scene.object_pose(key))
            items.append({"object": key, "pose": pose, "index": index, "xy": list(xy), "yaw_deg": yaw,
                          "dir": str(directory), "mask_px": int(frame.mask.sum())})
    return items


def run_fp(directories: list[str]) -> None:
    script = to_wsl_path(PROJECT_ROOT / "wsl" / "fp_run_dirs.sh")
    command = ["wsl.exe", "-d", WSL_DISTRO, "--", "bash", script, *(to_wsl_path(Path(d)) for d in directories)]
    subprocess.run(command, check=True)


def score(items: list[dict]) -> list[dict]:
    for item in items:
        directory = Path(item["dir"])
        truth = np.loadtxt(directory / "T_world_object_gt.txt")
        pose_file = directory / "T_cam_object.txt"
        if not pose_file.is_file():
            item["error"] = "no FoundationPose result"
            continue
        estimate = np.loadtxt(directory / "T_world_cam.txt") @ np.loadtxt(pose_file)
        spec = OBJECTS[item["object"]]
        keep = select_grasp(item["object"], truth).entry.keep
        axis = keep.axis if keep is not None else (0.0, 0.0, 1.0)
        translation, rotation = pose_error(estimate, truth, spec.symmetry, axis)
        item["translation_mm"] = round(translation * 1000.0, 2)
        item["rotation_deg"] = round(rotation, 2)
        # Does the estimate pick the same grasp (rest pose) as the truth would?
        try:
            item["grasp_matches"] = select_grasp(item["object"], estimate).name == select_grasp(item["object"], truth).name
        except RuntimeError:
            item["grasp_matches"] = False
        result = directory / "fp_result.json"
        if result.is_file():
            item.update({k: v for k, v in json.loads(result.read_text()).items() if k in ("register_s", "peak_gpu_mb")})
        item["pass"] = item["translation_mm"] < TRANSLATION_GATE_MM and item["rotation_deg"] < ROTATION_GATE_DEG
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    index_file = FRAMES / "index.json"
    if args.score_only:
        items = json.loads(index_file.read_text())
    else:
        items = export_frames(args.scenes, args.objects, args.seed)
        index_file.write_text(json.dumps(items, indent=2))
        print(f"exported {len(items)} frames to {FRAMES}")
        if args.export_only:
            return
        run_fp([item["dir"] for item in items])
    items = score(items)
    out = PROJECT_ROOT / "artifacts" / "benchmarks" / "pose_accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, indent=2))
    print(f"\n{'object':7s} {'pose':8s} {'n':>3s} {'pass':>5s} {'t med mm':>9s} {'t max mm':>9s} {'r med deg':>10s} {'r max deg':>10s} {'grasp ok':>9s} {'reg s':>6s}")
    for key, pose in CASES:
        group = [i for i in items if i["object"] == key and i["pose"] == pose and "translation_mm" in i]
        if not group:
            continue
        t = np.array([i["translation_mm"] for i in group])
        r = np.array([i["rotation_deg"] for i in group])
        register = [i["register_s"] for i in group if "register_s" in i]
        print(f"{key:7s} {pose:8s} {len(group):3d} {sum(i['pass'] for i in group):5d} {np.median(t):9.1f} {t.max():9.1f} "
              f"{np.median(r):10.1f} {r.max():10.1f} {sum(i['grasp_matches'] for i in group):9d} "
              f"{np.median(register) if register else float('nan'):6.2f}")
    missing = [i for i in items if "error" in i]
    if missing:
        print(f"{len(missing)} frame(s) without a FoundationPose result")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
