from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

from openarm_pick_place.models import Pose

# The lab's robot is OpenArm v1 (confirmed by the mentor, 2026-09-18). The v1 MJCF is
# vendored from enactic/openarm_mujoco (the pip package only ships v2) so the project
# is self-contained. Differences from v2 that matter here: the link chain runs along
# +z, joint6/joint7 are the x/y wrist bend axes (swapped vs v2), the tool axis is +z
# of link7 (no ee_base_link body), and the arm actuators are torque motors.
_PROJECT_ROOT = Path(os.environ.get("OPENARM_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
if not (_PROJECT_ROOT / "assets/openarm_v1/scene.xml").is_file():
    _PROJECT_ROOT = Path.cwd()
OPENARM_V1_DIR = _PROJECT_ROOT / "assets" / "openarm_v1"
MODEL_RELATIVE_PATH = Path("scene.xml")
LEFT_ARM_ACTUATORS = tuple(f"left_joint{joint}_ctrl" for joint in range(1, 8))
LEFT_EE_SITE = "left_ee_control_point"
# Where the stock gripper bolts on: link7's mesh ends at z=0.0955 in its own frame and
# the gripper's hand body starts there. This is the tool flange ("tool0") of v1.
FLANGE_Z = 0.0955
# Per motor class: servo gains plus the joint dynamics (damping, rotor armature,
# friction) the gains were tuned against. v1's MJCF has no armature and half the
# damping at a 2ms timestep, which makes the same gains diverge (joint2 ran to 190
# degrees, past its limit, within a second); the motors are physically the same parts.
SERVO_GAINS = {
    "motor_DM8009": dict(kp=200.0, kv=150.0, force=40.0, damping=1.0, armature=0.0081, frictionloss=0.2),
    "motor_DM4340": dict(kp=100.0, kv=30.0, force=27.0, damping=0.9, armature=0.16, frictionloss=0.1),
    "motor_DM4310": dict(kp=200.0, kv=5.0, force=7.0, damping=0.9, armature=0.01, frictionloss=0.04),
}
SIM_TIMESTEP = 0.001
ARM_JOINT_CLASSES = ("motor_DM8009", "motor_DM8009", "motor_DM4340", "motor_DM4340", "motor_DM4310", "motor_DM4310", "motor_DM4310")


def official_model_path() -> Path:
    """The lab robot's scene: OpenArm v1 bimanual with floor and lighting."""
    path = OPENARM_V1_DIR / MODEL_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"OpenArm v1 model missing at {path}; copy openarm_mujoco/v1 into assets/openarm_v1")
    return path


def configure_arm_servos(spec: mujoco.MjSpec) -> None:
    """(Re)write the seven arm actuators per side as position servos with SERVO_GAINS.

    Idempotent, and meant to be called again after any `attach_body`: attaching the
    Inspire hand merges that file's `<default><position kp="800">` into the spec and
    silently re-parametrises unclassed actuators (seen as kp 800 / kv 12 on joint2).
    """
    for side in ("left", "right"):
        for index, motor_class in enumerate(ARM_JOINT_CLASSES, start=1):
            name = f"{side}_joint{index}_ctrl"
            joint = spec.joint(f"openarm_{side}_joint{index}")
            gains = SERVO_GAINS[motor_class]
            # MuJoCo 3.13 exposes damping as a fixed-size array even for a
            # scalar hinge. Change only the hinge's active axis.
            joint.damping[0] = gains["damping"]
            joint.armature = gains["armature"]
            joint.frictionloss = gains["frictionloss"]
            actuator = spec.actuator(name)
            if actuator is None:
                actuator = spec.add_actuator(name=name)
            actuator.target = joint.name
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            # In-place: assigning a fresh array to these properties is silently ignored.
            actuator.gainprm[:] = 0.0
            actuator.gainprm[0] = gains["kp"]
            actuator.biasprm[:] = 0.0
            actuator.biasprm[1] = -gains["kp"]
            actuator.biasprm[2] = -gains["kv"]
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            actuator.forcerange = [-gains["force"], gains["force"]]
            actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            actuator.ctrlrange = list(np.asarray(joint.range, dtype=float))


def load_openarm_spec() -> mujoco.MjSpec:
    """OpenArm v1 as an MjSpec with position-servo arm actuators and a flange site per arm.

    The vendored MJCF drives the seven arm joints with torque motors; the planner and
    executor command joint *positions* (as ros2_control's JointTrajectoryController
    does on the real arm), so the motors are replaced by position actuators with the
    v2 gains and joint dynamics. A site named `<side>_ee_control_point` is added on
    each link7 at the flange face so the tool frame has one name across versions.
    """
    spec = mujoco.MjSpec.from_file(str(official_model_path()))
    spec.option.timestep = SIM_TIMESTEP
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    for side in ("left", "right"):
        for index in range(1, 8):
            old = spec.actuator(f"{side}_joint{index}_ctrl")
            if old is not None:
                spec.delete(old)
        flange = spec.body(f"openarm_{side}_link7")
        flange.add_site(name=f"{side}_ee_control_point", pos=[0.0, 0.0, FLANGE_Z], size=[0.006, 0, 0], rgba=[1, 0, 0, 1], group=4)
    configure_arm_servos(spec)
    return spec


class MujocoRobot:
    def __init__(self, model_path: str | Path | None = None, *, five_finger: bool = False) -> None:
        if five_finger:
            from simulation.five_finger_model import build_five_finger_model

            self.model = build_five_finger_model()
        elif model_path:
            self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        else:
            self.model = load_openarm_spec().compile()
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
