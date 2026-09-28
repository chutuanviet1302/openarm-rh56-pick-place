"""Grasp-library coverage: every object x rest pose x yaw, one physics trial each.

Runs the real pipeline (pose backend -> grasp library -> planner -> physics) on the
centre-basket layout, one subprocess per trial (a fresh MuJoCo each time, as
sweep_centre_basket.py does), and tabulates which object poses the right arm takes,
with the failure reason for the rest -- plan-time (no reachable jaw heading) versus
physics (grasp, carry, release).

    python -m scripts.sweep_grasp_library                     # all cases, gt poses
    python -m scripts.sweep_grasp_library --objects can --poses lying
    python -m scripts.sweep_grasp_library --backend foundationpose

Writes artifacts/benchmarks/grasp_library_<backend>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PLATFORM = 0.10
BASKET = (0.28, 0.0)
PICK = (0.28, -0.25)
# Yaw steps per rest pose: a round-from-above object needs one; a lying one is
# symmetric under 180 deg (the jaw may take it from either side), so 0..165.
CASES = {
    # The task object set (config/grasp_library.yaml): YCB 005, 007, 009, 017.
    ("can", "upright"): [0.0, 90.0],
    ("tuna_can", "upright"): [0.0, 90.0, 180.0],
    ("gelatin_box", "upright"): [float(y) for y in range(0, 180, 15)],  # box: symmetric under 180
    ("orange", "upright"): [0.0, 90.0, 180.0],
}
# Earlier coverage runs (lying can / pear, 2026-09-28) are in grasp_library_gt.json.
EXTRA_CASES = {
    ("can", "lying"): [float(y) for y in range(0, 180, 15)],
    ("apple", "upright"): [0.0, 90.0],
    ("pear", "lying"): [float(y) for y in range(0, 360, 30)],
}
TRIAL_TIMEOUT_S = 1200


def run_case(key: str, pose: str, yaw: float, backend: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "trial.json"
        command = [
            sys.executable, "-u", "-m", "simulation.pick_place_demo", "--headless", "--arm", "right",
            "--object", str(PICK[0]), str(PICK[1]), "--basket", str(BASKET[0]), str(BASKET[1]),
            "--platform", str(PLATFORM), "--pick-object", key, "--pick-pose", pose, "--pick-yaw", str(yaw),
            "--pose-backend", backend, "--report", str(report),
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=TRIAL_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return {"success": False, "failed_phase": "timeout", "failure_reason": f"> {TRIAL_TIMEOUT_S}s"}
        if not report.is_file():
            tail = (completed.stderr or completed.stdout).strip().splitlines()[-3:]
            return {"success": False, "failed_phase": "crash", "failure_reason": " | ".join(tail)}
        result = json.loads(report.read_text(encoding="utf-8"))[0]
        result["wall_seconds"] = round(time.perf_counter() - started, 1)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="gt")
    parser.add_argument("--objects", nargs="*", help="only these object keys")
    parser.add_argument("--poses", nargs="*", help="only these rest poses")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--extra", action="store_true", help="also the lying can, apple and pear")
    args = parser.parse_args()
    out = args.out or Path("artifacts") / "benchmarks" / f"task_objects_{args.backend}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    cases = {**CASES, **(EXTRA_CASES if args.extra else {})}
    for (key, pose), yaws in cases.items():
        if (args.objects and key not in args.objects) or (args.poses and pose not in args.poses):
            continue
        for yaw in yaws:
            result = run_case(key, pose, yaw, args.backend)
            row = {
                "object": key, "pose": pose, "yaw_deg": yaw, "success": bool(result.get("success")),
                "failed_phase": result.get("failed_phase"),
                "failure_reason": ((result.get("failure_reason") or "").splitlines() or [""])[0][:160] or None,
                "grasp_name": result.get("grasp_name"),
                "placement_error_mm": None if result.get("placement_error_m") is None else round(result["placement_error_m"] * 1000, 1),
                "final_tilt_deg": None if result.get("bottle_tilt_deg") is None else round(result["bottle_tilt_deg"], 1),
                "proof_lift_slip_mm": None if result.get("proof_lift_hand_rise_m") is None
                else round((result["proof_lift_hand_rise_m"] - result["proof_lift_rise_m"]) * 1000, 1),
                "grasp_forces": result.get("grasp_forces"),
                "wrist_bend_at_grasp_deg": result.get("wrist_pitch_at_grasp_deg"),
                "pose_error_mm": None if result.get("perception_error_m") is None else round(result["perception_error_m"] * 1000, 1),
                "pose_rotation_error_deg": result.get("pose_rotation_error_deg"),
                "wall_seconds": result.get("wall_seconds"),
            }
            rows.append(row)
            status = "PASS" if row["success"] else f"FAIL {row['failed_phase']}: {row['failure_reason']}"
            print(f"{key:7s} {pose:8s} yaw {yaw:5.0f}  {status}", flush=True)
            out.write_text(json.dumps({"backend": args.backend, "pick": PICK, "basket": BASKET,
                                       "platform_m": PLATFORM, "rows": rows}, indent=2), encoding="utf-8")

    print("\nsummary (object, pose): passed / tried")
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["object"], row["pose"]), []).append(row)
    for (key, pose), group in groups.items():
        passed = [r for r in group if r["success"]]
        planned = [r for r in group if r["failed_phase"] != "plan"]
        print(f"  {key:7s} {pose:8s} {len(passed):2d}/{len(group):2d} passed, {len(planned):2d} planned; "
              f"passing yaws {[r['yaw_deg'] for r in passed]}")
    print(f"total {sum(r['success'] for r in rows)}/{len(rows)}; report {out}")


if __name__ == "__main__":
    main()
