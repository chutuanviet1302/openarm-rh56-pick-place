from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

from openarm_pick_place.models import Pose

MODEL_RELATIVE_PATH = Path("v2/pedestal/bottle_scene.xml")
LEFT_ARM_ACTUATORS = tuple(f"left_joint{joint}_ctrl" for joint in range(1, 8))
LEFT_EE_SITE = "left_ee_control_point"


def official_model_path() -> Path:
    path = Path(sys.prefix) / "share" / "openarm_mujoco" / MODEL_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError("official OpenArm model not installed; run: pip install openarm-mujoco")
    return path


class MujocoRobot:
    def __init__(self, model_path: str | Path | None = None, *, five_finger: bool = False) -> None:
        if five_finger:
            from simulation.five_finger_model import build_five_finger_model

            self.model = build_five_finger_model()
        else:
            path = Path(model_path) if model_path else official_model_path()
            self.model = mujoco.MjModel.from_xml_path(str(path.resolve()))
        self.data = mujoco.MjData(self.model)
        self._actuators = np.array([self.model.actuator(name).id for name in LEFT_ARM_ACTUATORS])
        self._site = self.model.site(LEFT_EE_SITE).id
        self._hand_actuators = np.array(
            [index for index in range(self.model.nu) if (self.model.actuator(index).name or "").startswith("inspire_")],
            dtype=int,
        )

    def reset(self) -> None:
        home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def joint_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[self.model.joint(f"openarm_left_joint{i}").qposadr[0]] for i in range(1, 8)])

    def end_effector_pose(self) -> Pose:
        return Pose(self.data.site_xpos[self._site].copy(), self.data.site_xmat[self._site].reshape(3, 3).copy())

    def set_joint_targets(self, joints: np.ndarray) -> None:
        targets = np.asarray(joints, dtype=float)
        if targets.shape != (7,) or not np.all(np.isfinite(targets)):
            raise ValueError("left arm target must contain 7 finite joint values")
        limits = self.model.actuator_ctrlrange[self._actuators]
        if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]):
            raise ValueError("left arm target exceeds actuator limits")
        self.data.ctrl[self._actuators] = targets

    def set_hand_targets(self, joints: np.ndarray) -> None:
        targets = np.asarray(joints, dtype=float)
        if not self._hand_actuators.size:
            raise RuntimeError("five-finger hand is not loaded")
        if targets.shape != self._hand_actuators.shape or not np.all(np.isfinite(targets)):
            raise ValueError(f"hand target must contain {self._hand_actuators.size} finite joint values")
        limits = self.model.actuator_ctrlrange[self._hand_actuators]
        if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]):
            raise ValueError("hand target exceeds actuator limits")
        self.data.ctrl[self._hand_actuators] = targets

    def step(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("step duration must be positive")
        for _ in range(max(1, int(np.ceil(seconds / self.model.opt.timestep)))):
            mujoco.mj_step(self.model, self.data)

    def stop(self) -> None:
        self.data.ctrl[self._actuators] = self.joint_positions()
