"""Choose the safest arm route from actual IK and collision-checked plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from simulation.pick_place.planner import GraspPlanner, Plan
from simulation.pick_place.scene import Scene


class Route(str, Enum):
    DIRECT_RIGHT = "DIRECT_RIGHT"
    DIRECT_LEFT = "DIRECT_LEFT"
    HANDOFF_RIGHT_TO_LEFT = "HANDOFF_RIGHT_TO_LEFT"
    HANDOFF_LEFT_TO_RIGHT = "HANDOFF_LEFT_TO_RIGHT"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    source_arm: str | None
    target_arm: str | None
    reason: str
    plan: Plan | None = None


class TaskRouter:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene

    @staticmethod
    def _score(planner: GraspPlanner, plan: Plan) -> tuple[float, float]:
        margin = min(planner.joint_margin_degrees(q) for q in plan.joints.values())
        length = sum(
            float(np.linalg.norm(end - start))
            for path in plan.paths.values()
            for start, end in zip(path, path[1:])
        )
        return margin, -length

    def select(self, object_position: np.ndarray | None = None) -> RouteDecision:
        object_position = self.scene.object_position() if object_position is None else np.asarray(object_position, float)
        direct: list[tuple[tuple[float, float], str, Plan]] = []
        failures: list[str] = []
        for side in ("right", "left"):
            planner = GraspPlanner(self.scene, side)
            try:
                plan = planner.plan(object_position)
                direct.append((self._score(planner, plan), side, plan))
            except RuntimeError as error:
                failures.append(f"{side} direct: {str(error).splitlines()[0]}")
        if direct:
            _, side, plan = max(direct, key=lambda candidate: candidate[0])
            return RouteDecision(Route[f"DIRECT_{side.upper()}"], side, side, "direct IK and clearance checks passed", plan)

        pick_ok, place_ok = {}, {}
        for side in ("right", "left"):
            planner = GraspPlanner(self.scene, side)
            try:
                pick_ok[side] = planner.plan_pick(object_position)
            except RuntimeError as error:
                failures.append(f"{side} pick: {str(error).splitlines()[0]}")
            try:
                # Probe the actual set-down planner from a known reachable pick on
                # that arm's side; a pick probe at B is not equivalent to placing.
                probe_pick = (0.08, -0.38 if side == "right" else 0.38)
                probe = Scene(probe_pick, tuple(self.scene.basket_floor()[:2]))
                place_ok[side] = GraspPlanner(probe, side).plan(probe.object_position())
            except RuntimeError as error:
                failures.append(f"{side} place: {str(error).splitlines()[0]}")
        for source, target in (("right", "left"), ("left", "right")):
            if source in pick_ok and target in place_ok:
                route = Route[f"HANDOFF_{source.upper()}_TO_{target.upper()}"]
                return RouteDecision(route, source, target, "direct route unavailable; opposite arms pass pick/place preflight")
        return RouteDecision(Route.REJECTED, None, None, "; ".join(failures))
