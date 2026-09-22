"""Deterministic MuJoCo acceptance benchmark for one OpenArm side."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from simulation.pick_place import config as C
from simulation.pick_place.demo import Demo, run_trial, sample_layout


def load_fixture(path: Path, side: str) -> list[dict]:
    """Layouts from a saved fixture (scripts/make_benchmark_fixture.py)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["side"] != side:
        raise SystemExit(f"fixture {path} is for the {payload['side']} arm, not {side}")
    return payload["layouts"]


def benchmark(side: str, trials: int, seed: int, perception: bool, fixture: Path | None = None) -> dict:
    """Run `trials` episodes, from a saved fixture when given one and freshly drawn
    layouts otherwise. Prefer the fixture: IK convergence near the edge of the
    workspace is not reproducible run to run, so freshly drawn layouts move the pass
    rate for reasons that have nothing to do with the robot."""
    rng = np.random.default_rng(seed)
    layouts = load_fixture(fixture, side) if fixture else None
    if layouts is not None:
        trials = min(trials, len(layouts))
    results = []
    model_info = None
    for index in range(trials):
        if layouts is not None:
            entry = layouts[index]
            layout = {"pick_position": tuple(entry["pick_position"]), "basket_position": tuple(entry["basket_position"])}
            region = entry.get("region")
        else:
            layout = sample_layout(rng, side=side, perception=perception)
            region = None
        demo = Demo(**layout, side=side, perception=perception)
        if model_info is None:
            model_info = {
                "model_timestep_s": float(demo.model.opt.timestep),
                "arm_actuator_forcerange": demo.model.actuator_forcerange[demo.scene.arm_actuators[side]].tolist(),
            }
        result = run_trial(demo)
        del demo
        result.seed = seed
        result.trial_index = index
        row = asdict(result)
        if region is not None:
            row["region"] = region
        if perception and not result.success:
            oracle = run_trial(Demo(**layout, side=side, perception=False))
            row["failure_class"] = "perception" if oracle.success else "manipulation"
        results.append(row)
        print(f"[{index + 1:02d}/{trials}] {region or '':<6} {result.summary()}", flush=True)
    passed = sum(row["success"] for row in results)
    by_region: dict[str, dict[str, int]] = {}
    for row in results:
        if row.get("region"):
            tally = by_region.setdefault(row["region"], {"passed": 0, "trials": 0})
            tally["trials"] += 1
            tally["passed"] += bool(row["success"])
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mujoco_version": version("mujoco"),
        **model_info,
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest() if fixture else None,
        "side": side,
        "seed": seed,
        "fixture": str(fixture) if fixture else None,
        "by_region": by_region,
        "perception": perception,
        "trials": trials,
        "passed": passed,
        "pass_rate": passed / trials,
        "required_passes": int(np.ceil(0.95 * trials)),
        "thresholds": {
            "placement_error_m": C.PLACEMENT_ERROR_LIMIT_M,
            "tilt_deg": C.PROOF_LIFT_MAX_TILT_DEG,
            "penetration_m": max(C.TABLE_CONTACT_TOLERANCE, C.BASKET_CONTACT_TOLERANCE),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("right", "left"), required=True)
    parser.add_argument("--trials", type=int, default=C.BENCHMARK_TRIALS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--perception", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--fixture", type=Path,
        help="run this saved layout fixture instead of drawing layouts (recommended: "
             "benchmarks/fixtures/<arm>-<count>.json, built by scripts.make_benchmark_fixture)",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    fixture = args.fixture
    if fixture is not None and not fixture.exists():
        parser.error(f"no fixture at {fixture}; build one with scripts.make_benchmark_fixture")
    report = args.report or Path("artifacts/benchmarks") / f"{args.arm}-{args.seed}.json"
    payload = benchmark(args.arm, args.trials, args.seed, args.perception, fixture)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{payload['passed']}/{payload['trials']} passed -> {report}")
    if payload["passed"] < payload["required_passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
