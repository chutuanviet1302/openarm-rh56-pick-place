"""Pure kinematics: rotations, forward kinematics probes and the damped-least-squares IK.

Nothing here steps physics or touches a live MjData; every function works on a
throwaway MjData so it can be called from planning code without side effects.
"""

from __future__ import annotations

import mujoco
import numpy as np

from simulation.five_finger_model import HAND_PREFIX
from simulation.pick_place.config import (
    ARM_JOINTS,
    EE_SITE,
    IK_DAMPING,
    IK_MAX_ITERATIONS,
    IK_MAX_STEP,
    IK_POSITION_TOLERANCE,
    IK_ROTATION_TOLERANCE,
    IK_ROTATION_WEIGHT,
    NATURAL_GRASP_JOINTS,
    WRIST_BEND_INDICES,
    WRIST_STRAIGHT_GAIN,
)


def rotation_z(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])


def rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])


def quintic(tau: float) -> float:
    """Zero-jerk time scaling s(tau) for tau in [0, 1]: zero velocity *and* zero
    acceleration at both ends, unlike a plain cubic smoothstep."""
    tau = min(max(tau, 0.0), 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def orientation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """World-frame rotation vector (axis * angle) taking `current` to `desired`."""
    q_cur, q_des = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q_cur, current.reshape(9))
    mujoco.mju_mat2Quat(q_des, desired.reshape(9))
    q_cur_inv = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
    q_err = np.zeros(4)
    mujoco.mju_mulQuat(q_err, q_des, q_cur_inv)
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, q_err, 1.0)
    return vel


def upright_tilt_degrees(quaternion: np.ndarray) -> float:
    """Angle between the body's own z axis and world up.

    `2*acos(|w|)` is the *total* rotation angle, so a can merely turned about its own
    vertical axis (as it is when the hand yaws for the set-down) read as "tilted" by
    the full turn while standing perfectly upright.
    """
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
    up = rotation.reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))


def arm_indices(model: mujoco.MjModel, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(joint ids, qpos addresses, dof addresses) of one arm."""
    joint_ids = np.array([model.joint(name).id for name in ARM_JOINTS[side]])
    return joint_ids, model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]


def wrist_frame(model: mujoco.MjModel, side: str, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward kinematics only: (wrist position, wrist rotation) at the given arm joints."""
    data = mujoco.MjData(model)
    _, qpos_ids, _ = arm_indices(model, side)
    data.qpos[qpos_ids] = joints
    mujoco.mj_kinematics(model, data)
    site_id = model.site(EE_SITE[side]).id
    return data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy()


def natural_grasp_frame(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Wrist frame of the right hand at NATURAL_GRASP_JOINTS (the straight-wrist posture)."""
    return wrist_frame(model, "right", NATURAL_GRASP_JOINTS)


def solve_pose_ik(
    model: mujoco.MjModel,
    side: str,
    target: np.ndarray,
    target_mat: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    """Damped least-squares IK for the wrist site with a straight-wrist nullspace task.

    Raises RuntimeError when the pose is unreachable from `initial`; callers chain
    solutions (seed each waypoint from the previous one) to stay on one IK branch.
    """
    data = mujoco.MjData(model)
    joint_ids, qpos_ids, dof_ids = arm_indices(model, side)
    site_id = model.site(EE_SITE[side]).id
    data.qpos[qpos_ids] = initial
    jacobian_pos = np.zeros((3, model.nv))
    jacobian_rot = np.zeros((3, model.nv))
    n = len(qpos_ids)
    for _ in range(IK_MAX_ITERATIONS):
        mujoco.mj_forward(model, data)
        position_error = np.asarray(target) - data.site_xpos[site_id]
        rotation_error = orientation_error(data.site_xmat[site_id].reshape(3, 3), target_mat)
        if np.linalg.norm(position_error) < IK_POSITION_TOLERANCE and np.linalg.norm(rotation_error) < IK_ROTATION_TOLERANCE:
            return data.qpos[qpos_ids].copy()
        mujoco.mj_jacSite(model, data, jacobian_pos, jacobian_rot, site_id)
        jacobian = np.vstack((jacobian_pos[:, dof_ids], IK_ROTATION_WEIGHT * jacobian_rot[:, dof_ids]))
        error = np.concatenate((position_error, IK_ROTATION_WEIGHT * rotation_error))
        pseudo_inverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + IK_DAMPING * np.eye(6))
        update = pseudo_inverse @ error
        # Secondary task in the nullspace of the pose task: straighten the wrist.
        posture = np.zeros(n)
        for index in WRIST_BEND_INDICES:
            posture[index] = -WRIST_STRAIGHT_GAIN * data.qpos[qpos_ids[index]]
        update += (np.eye(n) - pseudo_inverse @ jacobian) @ posture
        step = np.linalg.norm(update)
        if step > IK_MAX_STEP:
            update *= IK_MAX_STEP / step
        data.qpos[qpos_ids] += update
        data.qpos[qpos_ids] = np.clip(data.qpos[qpos_ids], model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1])
    raise RuntimeError(f"IK failed for {side} target {np.asarray(target).tolist()}")


def hand_pose(model: mujoco.MjModel, side: str, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """(actuator ids, ctrl targets) of one Inspire hand, fully open or fully closed."""
    ids, targets = [], []
    for actuator_id in range(model.nu):
        name = model.actuator(actuator_id).name or ""
        if not name.startswith(f"{HAND_PREFIX}{side}_"):
            continue
        lower, upper = model.actuator_ctrlrange[actuator_id]
        ids.append(actuator_id)
        targets.append(upper if closed else lower)
    return np.asarray(ids), np.asarray(targets)
