"""Learned policy vs scripted waypoints on the same random layouts (held-out seed).

Each trial: one object at a random spot/heading (scripts/generate_grasp_demos.py's
sampler), perceive -> plan -> ready (both the same), then either the scripted
reach + grasp + proof lift or the policy's; stops after the grasp. Success = the object
rose >= 3 cm with the hand, no exception (collision, joint limit, failed grasp).

    python -m scripts.eval_grasp_policy --trials 50 --seed 1000
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.generate_grasp_demos import make_scene, sample_setup
from simulation.pick_place.demo import Demo
from simulation.pick_place.policy_demo import DEFAULT_POLICY, PolicyGraspDemo


def trial(method: str, setup, policy_path: Path) -> dict:
    kind, rest, xy, yaw = setup
    scene = make_scene(kind, rest, xy, yaw)
    kwargs = dict(scene=scene, side="right", pose_backend="gt", release="drop", place_offset=(0.0, -0.10))
    demo = PolicyGraspDemo(policy_path=policy_path, **kwargs) if method == "policy" else Demo(**kwargs)
    demo.grasp_retries = 0
    start_z = float(scene.object_position()[2])
    failure = None
    try:
        demo.run(stop_after="grasp")
    except RuntimeError as error:
        failure = str(error).splitlines()[0][:160]
    rise = float(scene.object_position()[2]) - start_z
    reach = next((e["t"] for e in demo.log.events if e["kind"] == "phase" and e["phase"] == "reach"), None)
    return {
        "method": method, "kind": f"{kind}/{rest}", "xy": list(xy), "yaw": yaw,
        "success": failure is None and rise >= 0.03, "failure": failure, "rise_cm": round(rise * 100, 2),
        "grasp_seconds": None if reach is None else round(float(scene.data.time) - reach, 2),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000, help="held out: demos use seeds 1, 2, ...")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--methods", nargs="+", default=["scripted", "policy"])
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "eval_grasp_policy.json")
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    setups = []
    while len(setups) < args.trials:
        setup = sample_setup(rng)
        try:
            make_scene(*setup)
        except ValueError:
            continue  # a spot the scene builder rejects, as in the demo generator
        setups.append(setup)
    results = []
    for index, setup in enumerate(setups):
        for method in args.methods:
            started = time.perf_counter()
            row = trial(method, setup, args.policy)
            results.append(row)
            print(f"[{index:3d}] {method:8s} {row['kind']:14s} {'OK  ' if row['success'] else 'FAIL'} "
                  f"rise {row['rise_cm']:5.1f} cm  {row['grasp_seconds']} s  {row['failure'] or ''} "
                  f"({time.perf_counter() - started:.0f}s wall)", flush=True)
        args.out.write_text(json.dumps(results, indent=1))  # keep partial results

    summary = defaultdict(lambda: defaultdict(list))
    for row in results:
        summary[row["method"]]["all"].append(row)
        summary[row["method"]][row["kind"]].append(row)
    print("\nmethod    group            success      grasp time (s, successes)")
    for method, groups in summary.items():
        for group, rows in sorted(groups.items()):
            ok = [r for r in rows if r["success"]]
            times = [r["grasp_seconds"] for r in ok if r["grasp_seconds"] is not None]
            print(f"{method:9s} {group:15s} {len(ok):3d}/{len(rows):<3d} {100 * len(ok) / len(rows):5.0f}%   "
                  f"{np.mean(times) if times else float('nan'):5.1f}")
    print(f"results: {args.out}")


if __name__ == "__main__":
    main()
