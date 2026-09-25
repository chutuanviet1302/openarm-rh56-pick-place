"""Robustness sweep of the centre-basket task: right hand places, left hand retrieves.

The simulation is deterministic -- the same pick point gives the same result every
run -- so "run each point many times" says nothing. Instead the pick point is drawn
at random (fixed seed) inside a disc around the nominal one, each trial runs both
legs with full physics, and the pass rate over the sample is the statistic.

    python -m scripts.sweep_centre_basket --n 40 --radius 0.015 --workers 4
    python -m scripts.sweep_centre_basket --n 40 --perception        # D435 camera
    -> artifacts/benchmarks/centre_basket_sweep_<label>.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

NOMINAL_PICK = (0.28, -0.25)
BASKET = (0.28, 0.0)
RETRIEVE_TO = (0.28, 0.25)
PLATFORM = 0.10


def sample_picks(n: int, radius: float, seed: int) -> list[tuple[float, float]]:
    """Uniform over the disc (sqrt radius), nominal point first."""
    rng = np.random.default_rng(seed)
    picks = [NOMINAL_PICK]
    while len(picks) < n:
        r = radius * np.sqrt(rng.uniform())
        a = rng.uniform(0.0, 2.0 * np.pi)
        picks.append((round(NOMINAL_PICK[0] + r * np.cos(a), 4), round(NOMINAL_PICK[1] + r * np.sin(a), 4)))
    return picks


def run_one(index: int, pick: tuple[float, float], perception: bool) -> dict:
    from simulation.pick_place.retrieve import RetrieveDemo, run_retrieve_trial

    started = time.perf_counter()
    log = io.StringIO()
    record = {"index": index, "pick": [float(v) for v in pick]}
    with contextlib.redirect_stdout(log):
        try:
            task = RetrieveDemo(pick, BASKET, RETRIEVE_TO, side="left", place_side="right",
                                work_platform_height=PLATFORM, perception=perception)
            result = run_retrieve_trial(task)
            record.update(
                success=bool(result.success),
                failure=result.failure_reason,
                placement_error_mm=round(float(result.placement_error_m) * 1000, 1),
                tilt_deg=round(float(result.bottle_tilt_deg), 1),
                perception_error_mm=None if result.perception_error_m is None else round(float(result.perception_error_m) * 1000, 1),
            )
        except Exception as error:  # a crash is a failed trial, not a crashed sweep
            record.update(success=False, failure=f"{type(error).__name__}: {error}")
    text = log.getvalue()
    record["grasp_attempts"] = text.count("GRASP")
    record["regrasps"] = text.count("rejected")
    record["perception_notes"] = [line.strip() for line in text.splitlines() if "object seen at" in line]
    record["wall_s"] = round(time.perf_counter() - started)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--radius", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--perception", action="store_true", help="locate the can with the simulated D435")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    picks = sample_picks(args.n, args.radius, args.seed)
    label = args.label or f"{'camera' if args.perception else 'truth'}_r{int(args.radius * 1000)}mm_n{args.n}_s{args.seed}"
    out = Path("artifacts") / "benchmarks" / f"centre_basket_sweep_{label}.json"
    results = []
    # A fresh process per trial: a reused worker kept each trial's MuJoCo renderers and
    # models alive, and 6 of them ran the machine out of memory mid-sweep (2026-09-25).
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run_one, i, p, args.perception) for i, p in enumerate(picks)]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            status = "PASS" if r["success"] else "FAIL"
            print(f"[{len(results):2d}/{len(picks)}] #{r['index']:2d} pick {r['pick']} {status} "
                  f"{r.get('placement_error_mm')}mm regrasps={r['regrasps']} {r.get('failure') or ''}", flush=True)
    results.sort(key=lambda r: r["index"])
    passed = sum(r["success"] for r in results)
    errors = [r["placement_error_mm"] for r in results if r["success"]]
    summary = {
        "label": label, "n": len(results), "passed": passed, "radius_m": args.radius, "seed": args.seed,
        "perception": args.perception, "nominal_pick": NOMINAL_PICK, "basket": BASKET, "retrieve_to": RETRIEVE_TO,
        "platform_m": PLATFORM,
        "placement_error_mm": {"median": float(np.median(errors)), "max": float(np.max(errors))} if errors else None,
        "trials": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"passed {passed}/{len(results)} -> {out}")


if __name__ == "__main__":
    main()
