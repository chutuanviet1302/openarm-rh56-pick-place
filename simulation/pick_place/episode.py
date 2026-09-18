"""Episode record and the console log.

`EpisodeLog` prints one timestamped line per phase/event and keeps every measured
value in a dict, so a failed trial can be read back from the report instead of
re-run with print statements added.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrialResult:
    """One demonstration episode, logged in full so a trial can be audited or used
    as training data without re-running it."""

    success: bool
    failure_reason: str | None
    failed_phase: str | None
    final_position: list[float]
    simulation_seconds: float
    placement_error_m: float = 0.0
    bottle_tilt_deg: float = 0.0
    contact_forces: dict[str, float] | None = None
    # Layout and perception
    pick_position: list[float] | None = None
    basket_position: list[float] | None = None
    perception_used: bool = False
    perceived_position: list[float] | None = None
    perception_error_m: float | None = None
    # Grasp evidence
    grasp_forces: dict[str, float] | None = None
    wrist_pitch_at_grasp_deg: float | None = None
    proof_lift_rise_m: float | None = None
    proof_lift_hand_rise_m: float | None = None
    proof_lift_tilt_deg: float | None = None
    carry_clearance_above_rim_m: float | None = None
    place_yaw_deg: float | None = None
    # Wrist target per phase (world frame) and the joint solution that reaches it
    phase_wrist_positions: dict[str, list[float]] | None = None
    phase_joint_targets: dict[str, list[float]] | None = None
    # Object position and wrist position observed at the end of each phase
    phase_observations: dict[str, dict[str, list[float]]] | None = None

    def summary(self) -> str:
        status = "PASS" if self.success else f"FAIL in {self.failed_phase}"
        line = (
            f"{status}  A={np.round(self.pick_position or [], 3).tolist()} B={np.round(self.basket_position or [], 3).tolist()} "
            f"placement {self.placement_error_m*1000:.1f}mm tilt {self.bottle_tilt_deg:.1f}deg"
        )
        if self.perception_error_m is not None:
            line += f" perception err {self.perception_error_m*1000:.1f}mm"
        if not self.success:
            line += f"  <- {(self.failure_reason or '')[:100]}"
        return line


def write_report(results: list[TrialResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")


class EpisodeLog:
    def __init__(self, sim_time, verbose: bool = False) -> None:
        self._sim_time = sim_time  # callable returning data.time
        self.verbose = verbose
        self.values: dict = {}
        self.current_phase: str | None = None
        self.observations: dict[str, dict[str, list[float]]] = {}
        # Everything printed, with its sim time, so a recording can show a timeline.
        self.events: list[dict] = []

    def _stamp(self) -> str:
        return f"[t={self._sim_time():6.2f}s]"

    def phase(self, index: int, total: int, name: str, message: str) -> None:
        self.current_phase = name
        self.events.append({"t": float(self._sim_time()), "kind": "phase", "phase": name, "message": message})
        print(f"{self._stamp()} {index}/{total} {name.upper()}: {message}")

    def note(self, message: str) -> None:
        self.events.append({"t": float(self._sim_time()), "kind": "note", "phase": self.current_phase, "message": message})
        print(f"{self._stamp()}      {message}")

    def debug(self, message: str) -> None:
        if self.verbose:
            print(f"{self._stamp()}      - {message}")

    def record(self, key: str, value) -> None:
        self.values[key] = value.tolist() if isinstance(value, np.ndarray) else value

    def observe(self, phase: str, **positions: np.ndarray) -> None:
        self.observations[phase] = {k: np.asarray(v).round(4).tolist() for k, v in positions.items()}
        if self.verbose:
            self.debug(f"after {phase}: " + ", ".join(f"{k}={np.round(v, 3).tolist()}" for k, v in positions.items()))
