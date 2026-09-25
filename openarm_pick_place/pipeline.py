from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable, Protocol

from .models import GraspConfig, ObjectPose, Pose, TrialResult, Workspace
from .motion import pick_place_waypoints, validate_target


class Robot(Protocol):
    def move_to(self, pose: Pose) -> bool: ...
    def stop(self) -> None: ...


class Hand(Protocol):
    def command(self, configuration: str, speed: float) -> bool: ...


class PickPlaceController:
    def __init__(self, robot: Robot, hand: Hand, workspace: Workspace, grasp: GraspConfig, place_pose: Pose):
        self.robot, self.hand = robot, hand
        self.workspace, self.grasp, self.place_pose = workspace, grasp, place_pose

    def run(self, object_pose: ObjectPose, *, max_age: float = 0.5) -> TrialResult:
        started = time.monotonic()
        result = TrialResult(details={"object_position_base": object_pose.pose.position.tolist()})
        try:
            if time.time() - object_pose.timestamp > max_age:
                raise ValueError("stale object pose")
            waypoints = pick_place_waypoints(object_pose.pose, self.place_pose, self.grasp)
            for pose in waypoints.values():
                validate_target(pose, self.workspace, table_height=0.0, clearance=0.0)
            steps: tuple[tuple[str, Callable[[], bool]], ...] = (
                ("PREGRASP", lambda: self.robot.move_to(waypoints["pregrasp"])),
                ("APPROACH", lambda: self.robot.move_to(waypoints["grasp"])),
                ("CLOSE", lambda: self.hand.command("BOTTLE_GRASP", 0.2)),
                ("LIFT", lambda: self.robot.move_to(waypoints["lift"])),
                ("TRANSFER", lambda: self.robot.move_to(waypoints["preplace"])),
                ("LOWER", lambda: self.robot.move_to(waypoints["place"])),
                ("RELEASE", lambda: self.hand.command("OPEN", 0.2)),
                ("RETREAT", lambda: self.robot.move_to(waypoints["retreat"])),
            )
            for state, operation in steps:
                result.final_state = state
                if not operation():
                    raise RuntimeError(f"{state.lower()} failed")
            result.success, result.final_state = True, "DONE"
        except (ValueError, RuntimeError) as error:
            result.failure_reason, result.final_state = str(error), "SAFE_ABORT"
            self.robot.stop()
        result.cycle_time = time.monotonic() - started
        return result


def append_trial(path: str | Path, result: TrialResult) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": time.time(),
        "success": result.success,
        "final_state": result.final_state,
        "failure_reason": result.failure_reason,
        "cycle_time": result.cycle_time,
        "details": json.dumps(result.details, separators=(",", ":")),
    }
    exists = destination.exists()
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
