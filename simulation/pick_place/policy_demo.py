"""A pick episode whose approach, grasp and proof lift come from the learned policy.

Perception (pose backend), the grasp library and the plan stay as in Demo: they give
the scripted READY pose above the object and, after the grasp, the carry and release.
Between the two, every CONTROL_PERIOD_S:

    observation (features.observation: joints, finger forces, wrist, perceived object)
      -> ChunkPolicy + temporal ensembling -> wrist target + hand commands
      -> MinkArm (QP differential IK) -> arm joint command; hand commands as they are

until the object has come up PROOF_RISE_M with the fingers pressing, or MAX_SECONDS
pass (a failed grasp, raised like the scripted proof lift's failures).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from simulation.pick_place import config as C
from simulation.pick_place.demo import Demo
from simulation.pick_place.kinematics import wrist_frame
from simulation.pick_place.mink_motion import MinkArm
from simulation.policy.features import FINGER_ORDER, apply_action, observation
from simulation.policy.model import ChunkPolicy, EnsembledController

CONTROL_PERIOD_S = 0.1
MAX_SECONDS = 14.0
PROOF_RISE_M = 0.03
TRACK_TOLERANCE_M = 0.001
DEFAULT_POLICY = Path("artifacts") / "policy" / "grasp_policy.pt"
_CACHE: dict[Path, ChunkPolicy] = {}


def load_policy(path: Path = DEFAULT_POLICY) -> ChunkPolicy:
    path = Path(path)
    if path not in _CACHE:
        _CACHE[path] = ChunkPolicy.load(path)
    return _CACHE[path]


class PolicyGraspDemo(Demo):
    def __init__(self, *, policy_path: Path = DEFAULT_POLICY, **kwargs) -> None:
        super().__init__(**kwargs)
        # A LeRobot checkpoint (a pretrained_model directory) or this repo's own .pt.
        self.policy_path = Path(policy_path)
        self.policy = None if self.policy_path.is_dir() else load_policy(self.policy_path)
        self.grasp_retries = 0  # the policy re-grasps by itself or fails
        self.policy_seconds: float | None = None

    def rest_pose(self) -> str:
        target = self.scene.grasp_target
        return "lying" if target is not None and "lying" in target.name else "upright"

    def phase_reach(self) -> None:
        scene, side, data = self.scene, self.side, self.data
        arm_group = f"{side}_arm"
        mink_arm = MinkArm(scene, side)
        if self.policy is None:
            from simulation.policy.lerobot_policy import LeRobotController

            controller = LeRobotController(self.policy_path)
        else:
            controller = EnsembledController(self.policy)
        kind, rest = scene.pick_type, self.rest_pose()
        start_z = float(scene.object_position()[2])
        started = float(data.time)
        light_grip = None
        # The commanded wrist pose, integrated from the actions exactly as the demos'
        # commands evolved; IK residuals then do not accumulate into it (taking the FK
        # of the arm command as the base each step, the command drifted 4 cm up while
        # a demo's own actions were replayed -- the oracle check).
        cmd_p, cmd_r = wrist_frame(self.model, side, data.ctrl[scene.arm_actuators[side]])
        command = np.eye(4)
        command[:3, :3], command[:3, 3] = cmd_r, cmd_p
        while float(data.time) - started < MAX_SECONDS:
            wrist = np.eye(4)
            wrist[:3, :3], wrist[:3, 3] = scene.wrist_rotation(side), scene.wrist_position(side)
            forces = scene.finger_contact_forces(side)
            obs = observation(data.qpos[scene.arm_qpos[side]], data.qpos[scene.hand_qpos[side]],
                              [forces.get(f, 0.0) for f in FINGER_ORDER], wrist, self.perceived_pose, kind, rest)
            # The action steps the commanded wrist pose (features.action).
            command, hand = apply_action(command, controller(obs))
            # Track it tightly: the posture term only keeps the arm where it is.
            q_now = data.qpos.copy()
            q_now[scene.arm_qpos[side]] = data.ctrl[scene.arm_actuators[side]]
            mink_arm.posture.set_target(q_now)
            arm_q = mink_arm.converge(command, iterations=150, tolerance_m=TRACK_TOLERANCE_M)
            data.ctrl[scene.hand_actuators[side]] = hand
            if light_grip is None and sum(f > 0.5 for f in forces.values()) >= 3:
                light_grip = data.ctrl[scene.hand_actuators[side]].copy()
            self.executor.move_to({arm_group: arm_q}, CONTROL_PERIOD_S)
            rise = float(scene.object_position()[2]) - start_z
            if rise >= PROOF_RISE_M and not self.fingers_not_pressing():
                break
        self.policy_seconds = float(data.time) - started
        self.light_grip_ctrl = light_grip if light_grip is not None else data.ctrl[scene.hand_actuators[side]].copy()
        rise = float(scene.object_position()[2]) - start_z
        self.log.record("proof_lift_rise_m", rise)
        self.log.record("policy_seconds", self.policy_seconds)
        self.log.note(f"policy grasp: object +{rise * 100:.1f} cm after {self.policy_seconds:.1f} s")
        if rise < PROOF_RISE_M:
            raise RuntimeError(f"policy grasp failed: object rose {rise * 100:.1f} cm in {self.policy_seconds:.1f} s")

    def phase_grasp(self) -> None:
        """Done inside phase_reach (the policy closes and lifts)."""
