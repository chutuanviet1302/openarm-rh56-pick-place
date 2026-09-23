"""Sequential (relay) handoff: one arm places the object at a staging point on the
table that both arms can reach directly, the other arm picks it up from there.

Simultaneous bimanual grasp (both hands on the object at once) was audited and
rejected: `scripts/check_handoff_geometry.py` finds no collision-free pose for the
RH56 pair on any candidate in its sampled workspace (best case 22.7mm inter-hand
penetration, see docs/PROJECT_REPORT.md). A relay keeps every grasp a single real
hand closing on the object -- no simultaneous contact, no assist -- at the cost of
one extra set-down/pick-up cycle.

This module is otherwise unproven in practice: `find_handoff_point` was swept over
the whole sampled workspace for the router's own opposite-workspace test layout and
found no usable staging point either, for a *different* reason than the collision
above -- placing (already holding the object) reaches to about y=0 (the table
centreline) at best, but freshly grasping needs roughly 30cm of clearance from it,
independent of x (see docs/PROJECT_REPORT.md, "Handoff" section, and
tests/test_bimanual_routing.py). No table point is both placeable-into by one arm
and pickable-from by the other. The code below is still correct and will pick up
any staging point that *does* exist (e.g. once the grasp geometry supports a
closer-in approach, or for a layout with less extreme reach), but as of this
writing no such layout is known.
"""

from __future__ import annotations

import numpy as np

from simulation.pick_place.demo import Demo, run_trial
from simulation.pick_place.episode import TrialResult
from simulation.pick_place.planner import GraspPlanner
from simulation.pick_place.scene import Scene

# Staging region: close to the torso centreline (clear of the pedestal riser, which
# spans roughly x in [-0.16, 0.10]) so a direct top-grasp from either arm's own side
# can reach it without crossing the body.
HANDOFF_X_RANGE = (0.16, 0.34)
HANDOFF_X_STEP = 0.03
HANDOFF_Y_RANGE = (-0.05, 0.05)
HANDOFF_Y_STEP = 0.025
# The object never lands exactly on the staging point (18mm off measured on the
# router's opposite-workspace layout, 2026-09-23), so the target arm must be able to
# pick it up anywhere within this radius of the point, not just at the point itself.
HANDOFF_LANDING_TOLERANCE = 0.02


def _leg_reachable(pick_position: tuple[float, float], place_position: tuple[float, float], side: str) -> bool:
    try:
        scene = Scene(pick_position, place_position)
    except ValueError:
        return False
    # Reference seeds only. The grasp seed bank reaches edge-of-workspace branches
    # that a relay then cannot carry on: on the router's opposite-workspace layout it
    # yielded staging points (0.22-0.255, 0.02-0.025) whose physics runs dropped the
    # can in the carry or missed the pick-up in 5 of 6 trials, at 0.6 and 0.3 rad/s
    # alike (2026-09-23). A relay chains two such legs; it is only offered on grasps
    # the reference posture reaches.
    try:
        GraspPlanner(scene, side, use_seed_bank=False).plan(scene.object_position())
    except RuntimeError:
        return False
    return True


def find_handoff_point(
    pick_position: tuple[float, float], basket_position: tuple[float, float], source: str, target: str,
) -> tuple[float, float]:
    """A table point the source arm can place at (picking from `pick_position`) and
    the target arm can then pick up from (placing at `basket_position`), verified by
    the same full plan (grasp, transfer, lower, collision) the executor will run.
    Raises RuntimeError with every candidate tried when none works."""
    failures = []
    for x in np.arange(HANDOFF_X_RANGE[0], HANDOFF_X_RANGE[1] + 1e-9, HANDOFF_X_STEP):
        for y in np.arange(HANDOFF_Y_RANGE[0], HANDOFF_Y_RANGE[1] + 1e-9, HANDOFF_Y_STEP):
            candidate = (float(x), float(y))
            if not _leg_reachable(pick_position, candidate, source):
                failures.append(f"{candidate}: {source} cannot place here")
                continue
            tol = HANDOFF_LANDING_TOLERANCE
            # 3x3 grid, diagonals included: the measured miss was diagonal (landed at
            # (0.200, 0.011) for a (0.22, 0.025) staging point).
            landings = [candidate] + [
                (candidate[0] + dx, candidate[1] + dy)
                for dx in (-tol, 0.0, tol) for dy in (-tol, 0.0, tol) if dx or dy
            ]
            missed = next((p for p in landings if not _leg_reachable(p, basket_position, target)), None)
            if missed is not None:
                failures.append(f"{candidate}: {target} cannot pick up from {tuple(round(v, 3) for v in missed)}")
                continue
            return candidate
    raise RuntimeError(f"no staging point reachable by both {source} and {target}:\n  " + "\n  ".join(failures))


class HandoffDemo:
    """Two chained single-arm episodes: `source` carries the object from
    `pick_position` to the staging point, `target` carries it on from there to
    `basket_position`. Each leg is an ordinary `Demo` -- same grasp, proof-lift and
    set-down checks as a direct route -- so nothing about the handoff is faked."""

    def __init__(
        self,
        pick_position: tuple[float, float],
        basket_position: tuple[float, float],
        *,
        source_arm: str,
        target_arm: str,
        handoff_position: tuple[float, float] | None = None,
        perception: bool = False,
        verbose: bool = False,
    ) -> None:
        if source_arm == target_arm:
            raise ValueError("source_arm and target_arm must differ for a handoff")
        self.pick_position = tuple(float(v) for v in pick_position)
        self.basket_position = tuple(float(v) for v in basket_position)
        self.source_arm = source_arm
        self.target_arm = target_arm
        self.perception = perception
        self.verbose = verbose
        self.route_reason = "direct route unavailable; relayed via staging point"
        self.handoff_position = (
            tuple(float(v) for v in handoff_position) if handoff_position is not None
            else find_handoff_point(self.pick_position, self.basket_position, source_arm, target_arm)
        )
        self.leg1 = Demo(self.pick_position, self.handoff_position, perception=perception, verbose=verbose, side=source_arm)
        self.leg2: Demo | None = None
        self.handoff_pose: np.ndarray | None = None

    def run(self, viewer=None, stop_after: str | None = None) -> None:
        """Run leg 1 (source arm, pick -> staging point) to completion, then leg 2
        (target arm, staging point -> basket). `stop_after` applies to leg 2 only:
        leg 1 always finishes, so the object is genuinely resting at the staging
        point -- not merely in flight -- before the target arm reaches for it."""
        self.leg1.run(viewer)
        self.handoff_pose = self.leg1.scene.object_position().copy()
        self.leg2 = Demo(
            (float(self.handoff_pose[0]), float(self.handoff_pose[1])), self.basket_position,
            perception=self.perception, verbose=self.verbose, side=self.target_arm,
        )
        self.leg2.run(viewer, stop_after=stop_after)


def run_handoff_trial(handoff: HandoffDemo, viewer=None, stop_after: str | None = None) -> TrialResult:
    """Run both legs as ordinary single-arm trials and combine them into one
    `TrialResult`. If leg 1 fails the object never reaches the staging point, so
    leg 2 never runs; the result then reports leg 1's own failure against the real
    basket position instead of the staging point it never got to hand off at."""
    route = f"HANDOFF_{handoff.source_arm.upper()}_TO_{handoff.target_arm.upper()}"
    leg1_result = run_trial(handoff.leg1, viewer, stop_after=None)
    if not leg1_result.success:
        leg1_result.route = route
        leg1_result.route_reason = handoff.route_reason
        leg1_result.source_arm = handoff.source_arm
        leg1_result.target_arm = handoff.target_arm
        leg1_result.failure_reason = f"leg1 ({handoff.source_arm} -> staging): {leg1_result.failure_reason}"
        leg1_result.basket_position = list(handoff.basket_position)
        final_xy = np.array(leg1_result.final_position[:2])
        leg1_result.placement_error_m = float(np.linalg.norm(final_xy - np.array(handoff.basket_position)))
        leg1_result.inside_basket = False
        return leg1_result

    handoff.handoff_pose = handoff.leg1.scene.object_position().copy()
    handoff.leg2 = Demo(
        (float(handoff.handoff_pose[0]), float(handoff.handoff_pose[1])), handoff.basket_position,
        perception=handoff.perception, verbose=handoff.verbose, side=handoff.target_arm,
    )
    leg2_result = run_trial(handoff.leg2, viewer, stop_after=stop_after)
    leg2_result.route = route
    leg2_result.route_reason = handoff.route_reason
    leg2_result.source_arm = handoff.source_arm
    leg2_result.target_arm = handoff.target_arm
    leg2_result.handoff_pose = handoff.handoff_pose.tolist()
    leg2_result.handoff_forces = leg2_result.grasp_forces
    leg2_result.pick_position = list(handoff.pick_position)
    leg2_result.simulation_seconds = float(leg1_result.simulation_seconds + leg2_result.simulation_seconds)
    leg2_result.max_penetration_m = float(leg1_result.max_penetration_m + leg2_result.max_penetration_m)
    if not leg2_result.success:
        leg2_result.failure_reason = f"leg2 ({handoff.target_arm} -> basket): {leg2_result.failure_reason}"
    return leg2_result
