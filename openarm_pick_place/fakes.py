from __future__ import annotations

from .models import Pose


class FakeRobot:
    def __init__(self) -> None:
        self.poses: list[Pose] = []
        self.stopped = False

    def move_to(self, pose: Pose) -> bool:
        self.poses.append(pose)
        return True

    def stop(self) -> None:
        self.stopped = True


class FakeHand:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def command(self, configuration: str, speed: float) -> bool:
        self.commands.append(configuration)
        return speed > 0
