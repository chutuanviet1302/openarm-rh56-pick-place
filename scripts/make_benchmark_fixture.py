"""Build the fixed benchmark fixture for one arm (plan stage 2).

The randomised acceptance test draws its layouts fresh from an RNG on every run,
which makes the pass rate move for reasons that have nothing to do with the code:
IK convergence near the edge of the workspace is not bit-reproducible, so a layout
that is accepted in one run is rejected in the next and the sample changes under
you. A fixture is drawn once, screened, written down and then reused, so a change
in the pass rate means a change in the robot.

Screening goes beyond "the planner solved it once". A layout is only kept if the
set-down still solves when the object sits somewhere other than the nominal jaw
centre, because it never sits exactly there: the grip shifts during the carry, and
a layout whose set-down only works for the nominal offset fails in physics with
"held-offset compensation unavailable" and drops the object short of the basket.

Layouts are labelled with the region they exercise, using what the planner already
reports rather than hand-drawn boxes:

    detour  the route has to bend around the pedestal (plan.route_strategy)
    edge    the grasp pose's tightest joint is in the bottom third for this arm
    centre  the rest

"Edge" is a third of the screened population rather than an absolute number of
degrees, because how much margin counts as comfortable is a property of this arm and
this task, not something to assert up front. Mapped across the reachable table the
tightest joint runs between roughly 2 and 12 degrees, so an absolute threshold either
catches everything or nothing.

Usage:
    python -m scripts.make_benchmark_fixture --arm right
    python -m scripts.make_benchmark_fixture --arm left --count 50 --seed 20260921
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.serve_viewer import parse_layout
from simulation.pick_place import config as C
from simulation.pick_place.planner import GraspPlanner
from simulation.pick_place.scene import Scene

# Measured by sweeping the table with the planner (see the module docstring): outside
# these the arm simply cannot get there, and sampling wider spends all its time on
# draws that the screen throws away. Given as magnitudes; the y sign follows the arm.
PICK_BOX = ((0.02, 0.27), (0.20, 0.43))
BASKET_BOX = ((0.19, 0.29), (0.01, 0.21))
PERCEPTION_ENVELOPE_M = 0.003  # measured head-camera error peaks near 2.6 mm


def _mirror(side: str) -> float:
    return 1.0 if side == "right" else -1.0


def _draw(rng: np.random.Generator, side: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """A candidate (object, basket) pair in the half-table the given arm works in."""
    sign = _mirror(side)
    pick = (float(rng.uniform(*PICK_BOX[0])), -sign * float(rng.uniform(*PICK_BOX[1])))
    basket = (float(rng.uniform(*BASKET_BOX[0])), -sign * float(rng.uniform(*BASKET_BOX[1])))
    return pick, basket


def screen(pick, basket, side: str, scene: Scene | None = None) -> dict | None:
    """Everything that must hold before a layout earns a place in the fixture.

    Returns the layout's record, or None with the reason printed. Kinematics only --
    no physics is run here, so this says "the arm can get there", not "the grasp holds".
    """
    try:
        parse_layout({"object": list(pick), "basket": list(basket)})
    except ValueError:
        return None
    try:
        if scene is None:
            scene = Scene(pick, basket)
        else:
            scene.model.body("pick_bottle").pos[:2] = pick
            scene.model.body("place_basket").pos[:2] = basket
            scene.pick_position, scene.basket_position = tuple(pick), tuple(basket)
            scene.reset()
        planner = GraspPlanner(scene, side)
        plan = planner.plan(scene.object_position())
    except (RuntimeError, ValueError):
        return None

    # The grip never lands on the nominal jaw centre. Require the set-down to survive
    # the object sitting off it, in every direction, by as much as it was seen to.
    fingers, thumb = planner.jaw_offsets(planner.orientation)
    nominal = 0.5 * (fingers + thumb)
    envelope = C.FIXTURE_HELD_OFFSET_ENVELOPE_M
    for dx, dy in ((envelope, 0.0), (-envelope, 0.0), (0.0, envelope), (0.0, -envelope)):
        try:
            planner.plan_place(
                copy.deepcopy(plan), scene.object_position(), held_offset=nominal + np.array([dx, dy, 0.0])
            )
        except RuntimeError:
            return None

    # A nominally reachable layout is not a useful perception benchmark when the
    # measured camera error alone pushes its plan outside the reachable set.
    for dx, dy in ((PERCEPTION_ENVELOPE_M, 0.0), (-PERCEPTION_ENVELOPE_M, 0.0),
                   (0.0, PERCEPTION_ENVELOPE_M), (0.0, -PERCEPTION_ENVELOPE_M)):
        try:
            planner.plan(scene.object_position() + np.array([dx, dy, 0.0]))
        except RuntimeError:
            return None

    # The grasp is the most constrained pose in the chain, and its margin actually
    # varies with the layout. The minimum across all phases does not: find_raise
    # returns the first waypoint over MIN_JOINT_MARGIN_DEG, so that number reports
    # where the accept test sits rather than how hard the layout is.
    margin = planner.joint_margin_degrees(plan.joints["grasp"])
    return {
        "pick_position": [round(v, 5) for v in pick],
        "basket_position": [round(v, 5) for v in basket],
        "route_strategy": plan.route_strategy,
        "grasp_yaw_deg": round(float(plan.grasp_yaw_deg), 1),
        "place_yaw_deg": round(float(plan.place_yaw_deg), 1),
        "grasp_joint_margin_deg": round(margin, 2),
        "pick_to_basket_m": round(float(np.hypot(basket[0] - pick[0], basket[1] - pick[1])), 4),
    }


def build(side: str, count: int, seed: int, max_attempts: int) -> dict:
    """Screen a pool of layouts, then pick the fixture out of it by region.

    Two passes rather than filling quotas as we draw: "edge" is defined against the
    margin the arm actually achieves, so the population has to exist before any layout
    can be labelled. Regions are then filled to an equal share so the fixture keeps
    exercising the hard cases; a shortfall is recorded rather than quietly padded.
    """
    quota = {region: count // 3 for region in ("centre", "edge", "detour")}
    for region in ("centre", "edge", "detour")[: count - sum(quota.values())]:
        quota[region] += 1

    rng = np.random.default_rng(seed)
    pool: list[dict] = []
    scene = Scene()
    for attempts in range(1, max_attempts + 1):
        record = screen(*_draw(rng, side), side=side, scene=scene)
        if record is not None:
            pool.append(record)
            print(f"  pooled {len(pool):3d} (draw {attempts})  margin "
                  f"{record['grasp_joint_margin_deg']:5.1f}deg  {record['route_strategy']}", flush=True)
        # Only the bottom third of the direct layouts can be labelled "edge", so
        # filling that share of the fixture needs three times as many directs pooled.
        directs = sum(1 for r in pool if r["route_strategy"] == "direct")
        detours = len(pool) - directs
        if directs >= 3 * quota["edge"] and detours >= quota["detour"]:
            break
    if not pool:
        raise SystemExit("no layout survived screening; check the sampling boxes")

    direct = [record for record in pool if record["route_strategy"] == "direct"]
    edge_cut = float(np.percentile([r["grasp_joint_margin_deg"] for r in direct], 100 / 3)) if direct else 0.0
    for record in pool:
        if record["route_strategy"] != "direct":
            record["region"] = "detour"
        else:
            record["region"] = "edge" if record["grasp_joint_margin_deg"] <= edge_cut else "centre"

    kept: dict[str, list[dict]] = {region: [] for region in quota}
    for record in pool:
        if len(kept[record["region"]]) < quota[record["region"]]:
            kept[record["region"]].append(record)

    layouts = [record for region in ("centre", "edge", "detour") for record in kept[region]]
    for index, record in enumerate(layouts):
        record["index"] = index
    shortfall = {region: quota[region] - len(kept[region]) for region in quota if len(kept[region]) < quota[region]}
    return {
        "pool_size": len(pool),
        "edge_margin_cut_deg": round(edge_cut, 2),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "side": side,
        "seed": seed,
        "count": len(layouts),
        "requested": count,
        "quota": quota,
        "achieved": {region: len(kept[region]) for region in quota},
        "shortfall": shortfall,
        "screen": {
            "held_offset_envelope_m": C.FIXTURE_HELD_OFFSET_ENVELOPE_M,
            "perception_envelope_m": PERCEPTION_ENVELOPE_M,
            "pick_box": PICK_BOX,
            "basket_box": BASKET_BOX,
        },
        "layouts": layouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("right", "left"), required=True)
    parser.add_argument("--count", type=int, default=C.BENCHMARK_TRIALS)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--max-attempts", type=int, default=600)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.count < 3:
        parser.error("--count must be at least 3 (one per region)")

    payload = build(args.arm, args.count, args.seed, args.max_attempts)
    out = args.out or Path("benchmarks/fixtures") / f"{args.arm}-{args.count}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    spread = Counter(record["region"] for record in payload["layouts"])
    print(f"{payload['count']} layouts ({dict(spread)}) from a pool of {payload['pool_size']} -> {out}")
    if payload["shortfall"]:
        print(f"short of quota: {payload['shortfall']}")


if __name__ == "__main__":
    main()
