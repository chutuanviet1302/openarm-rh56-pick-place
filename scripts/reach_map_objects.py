"""Where on the work platform can the right arm take each task object into the basket?

For every object (and heading) on a grid of footprint centres, alone on the table:
perceive (gt pose) -> grasp library -> full plan (grasp, lift, carry, drop over the
basket). One process per cell (fresh MuJoCo each time), several in parallel.

    python -m scripts.reach_map_objects                # all task objects, 4 workers
    python -m scripts.reach_map_objects --workers 2 --objects tuna_can

Writes artifacts/benchmarks/reach_map_objects.json and prints a map per object
(# = plans, . = does not); the table-clearing layout is chosen from it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

XS = (0.22, 0.26, 0.30, 0.34, 0.38, 0.42)
YS = (-0.19, -0.23, -0.27, -0.31, -0.35, -0.39, -0.43)
HEADINGS = {"can": (0.0,), "orange": (0.0,), "tuna_can": (0.0,), "gelatin_box": (0.0, 120.0)}
CELL_TIMEOUT_S = 900

CELL = r"""
import sys
from simulation.pick_place.scene import Scene
from simulation.pick_place.demo import Demo
key, x, y, yaw = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
try:
    scene = Scene((x, y), (0.28, 0.0), work_platform_height=0.10, pick_object=key, pick_yaw_deg=yaw)
    demo = Demo(scene=scene, side="right", pose_backend="gt", release="drop", place_offset=(0.0, 0.0))
    demo.phase_perceive()
    demo.phase_plan()
    bend = abs(float(demo.plan.joints["grasp"][5])) + abs(float(demo.plan.joints["grasp"][6]))
    print("CELL OK", demo.scene.grasp_target.name, round(bend, 3))
except Exception as error:
    print("CELL NO", str(error).splitlines()[0][:120])
"""


def run_cell(key: str, x: float, y: float, yaw: float) -> dict:
    started = time.perf_counter()
    try:
        out = subprocess.run([sys.executable, "-c", CELL, key, str(x), str(y), str(yaw)],
                             capture_output=True, text=True, timeout=CELL_TIMEOUT_S).stdout
    except subprocess.TimeoutExpired:
        out = "CELL NO timeout"
    line = next((l for l in out.splitlines() if l.startswith("CELL")), "CELL NO crashed")
    ok = line.startswith("CELL OK")
    return {"object": key, "x": x, "y": y, "yaw": yaw, "ok": ok, "detail": line[8:],
            "seconds": round(time.perf_counter() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--objects", nargs="*", default=list(HEADINGS))
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "benchmarks" / "reach_map_objects.json")
    args = parser.parse_args()
    jobs = [(k, x, y, yaw) for k in args.objects for yaw in HEADINGS[k] for x in XS for y in YS]
    print(f"{len(jobs)} cells, {args.workers} workers", flush=True)
    rows = []
    with ThreadPoolExecutor(args.workers) as pool:
        for row in pool.map(lambda job: run_cell(*job), jobs):
            rows.append(row)
            print(f"{row['object']:12s} yaw {row['yaw']:5.0f} ({row['x']:.2f}, {row['y']:.2f}) "
                  f"{'OK' if row['ok'] else '--'} {row['seconds']:5.0f}s {row['detail'][:70]}", flush=True)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"xs": XS, "ys": YS, "rows": rows}, indent=1))
    for key in args.objects:
        for yaw in HEADINGS[key]:
            print(f"\n{key} yaw {yaw:.0f}   (rows x, columns y {YS})")
            for x in XS:
                cells = {r["y"]: r["ok"] for r in rows if r["object"] == key and r["yaw"] == yaw and r["x"] == x}
                print(f"  x {x:.2f}  " + " ".join("#" if cells.get(y) else "." for y in YS))


if __name__ == "__main__":
    main()
