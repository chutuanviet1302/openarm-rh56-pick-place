"""Choose the safest arm route from actual IK and collision-checked plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from simulation.pick_place.handoff import find_handoff_point
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
    handoff_position: tuple[float, float] | None = None


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

        pick_ok = {}
        for side in ("right", "left"):
            try:
                pick_ok[side] = GraspPlanner(self.scene, side).plan_pick(object_position)
            except RuntimeError as error:
                failures.append(f"{side} pick: {str(error).splitlines()[0]}")
        pick_xy = (float(object_position[0]), float(object_position[1]))
        basket_xy = tuple(self.scene.basket_position)
        for source, target in (("right", "left"), ("left", "right")):
            if source not in pick_ok:
                continue
            try:
                handoff_xy = find_handoff_point(pick_xy, basket_xy, source, target)
            except RuntimeError as error:
                failures.append(f"{source}->{target} handoff: {str(error).splitlines()[0]}")
                continue
            route = Route[f"HANDOFF_{source.upper()}_TO_{target.upper()}"]
            return RouteDecision(
                route, source, target,
                f"direct route unavailable; relayed via staging point {tuple(round(v, 3) for v in handoff_xy)}",
                handoff_position=handoff_xy,
            )
        return RouteDecision(Route.REJECTED, None, None, "; ".join(failures))
