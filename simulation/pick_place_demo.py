"""Compatibility entry point. The implementation lives in `simulation/pick_place/`;
this module re-exports the public names so existing imports and
`python -m simulation.pick_place_demo` keep working."""

from __future__ import annotations

from simulation.pick_place.cli import main
from simulation.pick_place.config import *  # noqa: F401,F403
from simulation.pick_place.demo import PHASES, Demo, object_to_basket_distance, run_trial, sample_layout
from simulation.pick_place.episode import TrialResult, write_report
from simulation.pick_place.kinematics import (
    natural_grasp_frame,
    orientation_error,
    quintic,
    rotation_y,
    rotation_z,
    solve_pose_ik,
    upright_tilt_degrees,
)
from simulation.pick_place.planner import PHASE_ORDER, GraspPlanner, Plan
from simulation.pick_place.scene import Scene
from simulation.pick_place.executor import Executor

if __name__ == "__main__":
    main()
