from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

from simulation.five_finger_model import HAND_PREFIX, build_five_finger_model

ARM_JOINTS = {
    "left": tuple(f"openarm_left_joint{i}" for i in range(1, 8)),
    "right": tuple(f"openarm_right_joint{i}" for i in range(1, 8)),
}
ARM_ACTUATORS = {
    "left": tuple(f"left_joint{i}_ctrl" for i in range(1, 8)),
    "right": tuple(f"right_joint{i}_ctrl" for i in range(1, 8)),
}
EE_SITE = {"left": "left_ee_control_point", "right": "right_ee_control_point"}
BOTTLE_JOINT = "pick_bottle_joint"

# The grasp is no longer a hand-tuned table of wrist positions. Those were all placed
# by eye above the bottle's waist, and every one of them tipped the bottle over: the
# YCB mustard bottle is 23cm tall, so any squeeze applied above its centre of mass
# (z 0.515) rotates it instead of holding it -- measured 90 degrees of roll on every
# variant tried. The wrist targets are now *derived* at runtime from the bottle's own
# centre of mass and the hand's measured finger offset, so the fingers land on the
# bottle's middle by construction.
PHASE_ORDER = ("hover", "ready", "pregrasp", "grasp", "lift", "transfer", "lower")
# Tilt of the palm away from horizontal, about the wrist's y axis. 0 points the
# fingers straight out (the arm cannot reach the bottle's waist that way -- the wrist
# cannot get below z 0.70), 90 is fully palm-down (the thumb then has to cross the
# bottle to oppose, so the approach always collides). 45 was the only region that
# reached the centre of mass *and* approached without touching the bottle.
GRASP_TILT_DEGREES = 60.0
GRASP_CLOSURE_FRACTION = 0.20
# How far outside the bottle's far face the open fingertips sit on arrival, and a
# small extra bias measured to keep the approach contact-free.
GRASP_CLEARANCE = 0.0005
GRASP_Y_BIAS = -0.012
# The hand backs off along its own finger axis, so the approach re-enters the way the
# jaw points and the bottle slides into the slot rather than a jaw sweeping through it.
# This has to clear the bottle's whole 9.6cm depth plus the hand's own span: at 0.08
# the fingers were still buried in the bottle at the standoff pose (measured 11.6N on
# the index finger before the approach even started).
APPROACH_STANDOFF = 0.10
# Raises the grip so the fingers close around the object rather than into the table.
GRASP_HEIGHT_BIAS = 0.03
GRASP_POSITION_CORRECTION = np.array([0.06, -0.08, 0.0])
# 0 sits the wrist at the jaw midpoint, 0.5 puts the fingers themselves on the object.
JAW_BIAS_TOWARD_FINGERS = 0.0
# How high the hand rides before descending onto the standoff.
HOVER_HEIGHT = 0.08
# A contact only counts as a crash once it is deeper than this. MuJoCo reports a contact
# slightly before the surfaces actually interpenetrate, so an exact-zero threshold aborted
# runs on a 0.2mm graze of the thumb against the table -- a touch, not a collision.
TABLE_CONTACT_TOLERANCE = 0.003
# The resting stance must clear the table by this much, not merely avoid touching it:
# the position servos sag under gravity while the robot holds still at the start.
ATTENTION_TABLE_CLEARANCE = 0.05
LIFT_HEIGHT = 0.10
# Test lift used to prove the grasp before committing to the carry.
PROOF_LIFT_HEIGHT = 0.05
PROOF_LIFT_MIN_RISE = 0.04
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
# Adaptive-closing parameters, following the force phase in
# correlllab/rh56_controller (grasp_executor._adaptive_force_phase): step each
# finger toward closure a little at a time and stop commanding a finger once it
# has pressed hard enough, so the fingers settle ON the surface instead of being
# driven through it.
CONTACT_FORCE_TARGET_N = 0.4
# Small steps with a long settle between them: a coarse step lets a finger build up
# a large penetration in one physics-free jump, and the solver then pushes the bottle
# away rather than the finger stopping on it. Measured slide fell from 17cm to ~5cm
# going 0.06 -> 0.012 -> 0.004 here (the rest of the fix was the deeper wrist x).
CLOSE_STEP_FRACTION = 0.004
CLOSE_SETTLE_SECONDS = 0.05
CLOSE_MAX_ITERATIONS = 260
LEFT_SEED = np.array([-0.587, -0.116, 0.738, 0.202, -1.003, -0.528, -1.460])
RIGHT_SEED = np.array([0.587, -0.116, -0.738, 0.202, 1.003, -0.528, 1.460])

# Full target orientation for the top-down reach, identical for both hands (their meshes
# are already opposite-chirality, so one shared world-frame target lands mirrored and
# consistent instead of each side's IK picking its own free yaw about the approach axis --
# which is what previously left the two hands twisted at unrelated, wrong-looking angles).
# Local Y = world down (the approach axis); X/Z fix the remaining yaw.
DOWN_ORIENTATION = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)


def rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    return np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])


# Palm tilted 45 degrees: fingers reach down and forward, the way a person reaches
# across a table for a bottle. Fully palm-down (90) puts the fingertips 21cm below the
# wrist, and the arm cannot then get them down to the bottle's waist; fully horizontal
# (0) cannot reach it either. See GRASP_TILT_DEGREES.
SIDE_GRASP_ORIENTATION = rotation_y(GRASP_TILT_DEGREES) @ DOWN_ORIENTATION


def quintic(tau: float) -> float:
    """Zero-jerk time scaling s(tau) for tau in [0, 1]: zero velocity *and*
    zero acceleration at both ends, unlike a plain cubic smoothstep."""
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


def solve_pose_ik(model: mujoco.MjModel, side: str, target: np.ndarray, target_mat: np.ndarray, initial: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    joint_ids = np.array([model.joint(name).id for name in ARM_JOINTS[side]])
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    site_id = model.site(EE_SITE[side]).id
    data.qpos[qpos_ids] = initial
    jacobian_pos = np.zeros((3, model.nv))
    jacobian_rot = np.zeros((3, model.nv))
    for _ in range(6000):
        mujoco.mj_forward(model, data)
        current_rotation = data.site_xmat[site_id].reshape(3, 3)
        position_error = np.asarray(target) - data.site_xpos[site_id]
        rotation_error = orientation_error(current_rotation, target_mat)
        if np.linalg.norm(position_error) < 0.006 and np.linalg.norm(rotation_error) < 0.05:
            return data.qpos[qpos_ids].copy()
        mujoco.mj_jacSite(model, data, jacobian_pos, jacobian_rot, site_id)
        jacobian_arm = np.vstack((jacobian_pos[:, dof_ids], 0.4 * jacobian_rot[:, dof_ids]))
        error = np.concatenate((position_error, 0.4 * rotation_error))
        update = jacobian_arm.T @ np.linalg.solve(jacobian_arm @ jacobian_arm.T + 0.004 * np.eye(6), error)
        step = np.linalg.norm(update)
        if step > 0.08:
            update *= 0.08 / step
        data.qpos[qpos_ids] += update
        data.qpos[qpos_ids] = np.clip(data.qpos[qpos_ids], model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1])
    raise RuntimeError(f"IK failed for {side} target {target.tolist()}")


def hand_pose(model: mujoco.MjModel, side: str, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    ids, targets = [], []
    for actuator_id in range(model.nu):
        name = model.actuator(actuator_id).name or ""
        if not name.startswith(f"{HAND_PREFIX}{side}_"):
            continue
        lower, upper = model.actuator_ctrlrange[actuator_id]
        target = lower
        if closed:
            target = upper if name.endswith(("thumb_proximal", "thumb_yaw")) else np.clip(0.5, lower, upper)
        ids.append(actuator_id)
        targets.append(np.clip(target, lower, upper))
    return np.asarray(ids), np.asarray(targets)


@dataclass
class TrialResult:
    success: bool
    failure_reason: str | None
    final_position: list[float]
    simulation_seconds: float


class Demo:
    # See _right_grasp_is_secure(): True demands a real opposed thumb+fingers pinch;
    # False (current) accepts a secure multi-point finger wrap because the thumb does
    # not yet reach the bottle with this grasp approach.
    REQUIRE_THUMB_OPPOSITION = True

    def __init__(self) -> None:
        self.model = build_five_finger_model(pick_bottle=True)
        self.data = mujoco.MjData(self.model)

        self.arm_qpos, self.arm_dofs, self.arm_actuators = {}, {}, {}
        self.hand_qpos, self.hand_dofs, self.hand_actuators = {}, {}, {}
        self.open_hand, self.closed_hand = {}, {}
        for side in ("left", "right"):
            joint_ids = np.array([self.model.joint(name).id for name in ARM_JOINTS[side]])
            self.arm_qpos[side] = self.model.jnt_qposadr[joint_ids]
            self.arm_dofs[side] = self.model.jnt_dofadr[joint_ids]
            self.arm_actuators[side] = np.array([self.model.actuator(name).id for name in ARM_ACTUATORS[side]])
            hand_actuators, open_targets = hand_pose(self.model, side, False)
            _, closed_targets = hand_pose(self.model, side, True)
            hand_joints = self.model.actuator_trnid[hand_actuators, 0]
            self.hand_actuators[side] = hand_actuators
            self.hand_qpos[side] = self.model.jnt_qposadr[hand_joints]
            self.hand_dofs[side] = self.model.jnt_dofadr[hand_joints]
            self.open_hand[side] = open_targets
            self.closed_hand[side] = closed_targets

        # Per-finger actuator ids and their open/closed ctrl values, so the adaptive
        # closing loop can command and hold one finger at a time.
        self.finger_actuator: dict[str, dict[str, int]] = {}
        self.open_ctrl: dict[str, dict[str, float]] = {}
        self.closed_ctrl: dict[str, dict[str, float]] = {}
        self.thumb_yaw: dict[str, tuple[int, float]] = {}
        for side in ("left", "right"):
            self.finger_actuator[side] = {}
            self.open_ctrl[side] = {}
            self.closed_ctrl[side] = {}
            for position, actuator in enumerate(self.hand_actuators[side]):
                name = self.model.actuator(int(actuator)).name or ""
                tail = name[len(f"{HAND_PREFIX}{side}_") :]
                lower, upper = self.model.actuator_ctrlrange[int(actuator)]
                if tail.startswith("thumb_yaw"):
                    # Opposition (abduction), not a closing DOF. 0 swings the thumb out
                    # beside the palm (jaw opens to ~9.8cm); 1.308 brings it across the
                    # palm to face the fingers (jaw ~6.8cm). It is set once as a
                    # pre-shape before the approach -- the same "position the thumb
                    # first, then close" ordering as the reference hand's thumb reflex
                    # (rh56_controller grasp_executor._run_thumb_reflex).
                    self.thumb_yaw[side] = (int(actuator), float(lower), float(upper))
                    continue
                if tail.startswith("thumb_proximal"):
                    # Flexion. This is the thumb's closing DOF; joint equalities carry
                    # it through to the intermediate and distal knuckles.
                    self.finger_actuator[side]["thumb"] = int(actuator)
                    self.open_ctrl[side]["thumb"] = float(lower)
                    self.closed_ctrl[side]["thumb"] = float(upper)
                    continue
                if tail.startswith("thumb"):
                    continue
                self.finger_actuator[side][tail] = int(actuator)
                self.open_ctrl[side][tail] = float(self.open_hand[side][position])
                # Closing drives all the way to full curl and lets contact force stop
                # the finger, rather than stopping at a fixed half-closed angle that
                # may never reach the object (the reference force phase does the same:
                # it steps toward *min_closure*, not toward the planned angle).
                self.closed_ctrl[side][tail] = float(upper)

        self.bottle_qpos = self.model.joint(BOTTLE_JOINT).qposadr[0]
        self.bottle_dof = self.model.joint(BOTTLE_JOINT).dofadr[0]
        self.bottle_body = self.model.body("pick_bottle").id
        self.ee_site_id = {side: self.model.site(EE_SITE[side]).id for side in ("left", "right")}
        self.palm_body = {side: self.model.body(f"{HAND_PREFIX}{side}_base").id for side in ("left", "right")}

        home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
        # _phase_centers() reads the bottle's and basket's world positions, so the
        # scene has to be posed before the targets are derived from it.
        mujoco.mj_forward(self.model, self.data)
        self.poses = {}
        self.attention_pose = self._attention_pose()
        for side in ("left", "right"):
            self.data.qpos[self.arm_qpos[side]] = self.attention_pose[side]
            self.data.ctrl[self.arm_actuators[side]] = self.attention_pose[side]
            self.data.qpos[self.hand_qpos[side]] = self.closed_hand[side]
            self.data.ctrl[self.hand_actuators[side]] = self.closed_hand[side]
        mujoco.mj_forward(self.model, self.data)

    def _attention_pose(self) -> dict[str, np.ndarray]:
        """Arms at the sides, fists closed -- bent enough to clear the table properly.

        All-zero joints give a fully straight arm hanging at the shoulder's resting
        line, which is the stance we want. It only works while the shoulders are high
        above the table: at the pedestal height needed to reach the can, straight arms
        hang *below* the table top. Picking merely the first bend with no contact is not
        enough either -- the position servos sag under gravity during the opening hold
        and the hand settles onto the table anyway -- so this asks for real clearance.
        """
        start = np.zeros(7)
        start[3] = 1.5
        poses = {}
        for side in ("left", "right"):
            self.data.qpos[self.arm_qpos[side]] = start
        mujoco.mj_forward(self.model, self.data)
        for side in ("left", "right"):
            site = self.ee_site_id[side]
            poses[side] = solve_pose_ik(
                self.model,
                side,
                self.data.site_xpos[site] + np.array([0.0, 0.0, ATTENTION_TABLE_CLEARANCE + 0.03]),
                self.data.site_xmat[site].reshape(3, 3),
                start,
            )
        return poses

    def _jaw_offsets(self, orientation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean fingertip and thumb-tip positions relative to the wrist, in world axes.

        Measured by forward kinematics on a throwaway pose with the hand pre-shaped,
        so the grasp targets follow from where this particular hand's jaws actually
        end up rather than from numbers typed into a table.
        """
        # The tips sit at a fixed offset in the *wrist's own* frame once the hand is
        # pre-shaped, so this needs no IK at all: measure the offsets once from any
        # arm pose, express them in the wrist frame, then rotate into the requested
        # orientation. Probing with IK instead made this fail outright whenever the
        # probe point happened to be unreachable, which had nothing to do with the
        # hand geometry being asked about.
        data = mujoco.MjData(self.model)
        for name, actuator in self.finger_actuator["right"].items():
            joint = self.model.actuator_trnid[actuator, 0]
            opened = self.open_ctrl["right"][name]
            closed = self.closed_ctrl["right"][name]
            data.qpos[self.model.jnt_qposadr[joint]] = opened + GRASP_CLOSURE_FRACTION * (closed - opened)
        yaw_actuator, _, opposed = self.thumb_yaw["right"]
        yaw_joint = self.model.actuator_trnid[yaw_actuator, 0]
        data.qpos[self.model.jnt_qposadr[yaw_joint]] = opposed
        for equality in range(self.model.neq):
            if self.model.eq_type[equality] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            driven = int(self.model.eq_obj1id[equality])
            source = int(self.model.eq_obj2id[equality])
            if not (self.model.joint(driven).name or "").startswith(f"{HAND_PREFIX}right_"):
                continue
            source_value = data.qpos[self.model.jnt_qposadr[source]]
            coefficients = self.model.eq_data[equality, :5]
            value = sum(float(coefficient) * source_value**power for power, coefficient in enumerate(coefficients))
            data.qpos[self.model.jnt_qposadr[driven]] = np.clip(value, *self.model.jnt_range[driven])
        mujoco.mj_forward(self.model, data)

        site = self.ee_site_id["right"]
        wrist = data.site_xpos[site].copy()
        wrist_rotation = data.site_xmat[site].reshape(3, 3)
        tips = [
            data.site_xpos[self.model.site(f"{HAND_PREFIX}right_right_{finger}_tip").id]
            for finger in ("index", "middle", "ring", "pinky")
        ]
        thumb = data.site_xpos[self.model.site(f"{HAND_PREFIX}right_right_thumb_tip").id]
        local_fingers = wrist_rotation.T @ (np.mean(tips, axis=0) - wrist)
        local_thumb = wrist_rotation.T @ (thumb - wrist)
        return orientation @ local_fingers, orientation @ local_thumb


    def _object_extents(self) -> tuple[float, float]:
        """(width across the grasp, full height) of the object's collision geom.

        MuJoCo packs geom_size differently per type -- a cylinder stores
        (radius, half-height) while a box stores three half-extents -- so reading
        size[1] blindly returned the can's *height* as its width and tripped the
        jaw-clearance check against the wrong number entirely.
        """
        geom = self.model.geom("pick_bottle_collision")
        size = np.asarray(geom.size)
        if geom.type[0] in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            return 2.0 * float(size[0]), 2.0 * float(size[1])
        if geom.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE:
            return 2.0 * float(size[0]), 2.0 * float(size[0])
        return 2.0 * float(size[1]), 2.0 * float(size[2])

    def _phase_centers(self) -> dict[str, np.ndarray]:
        """Wrist targets derived from the bottle, the basket and the hand's geometry."""
        bottle = self.data.qpos[self.bottle_qpos : self.bottle_qpos + 3].copy()
        width, height = self._object_extents()
        offset, thumb_offset = self._jaw_offsets(SIDE_GRASP_ORIENTATION)
        axis = offset / np.linalg.norm(offset)

        # Straddle the object with the jaw centred on it in all three axes. Putting the
        # *fingers* on the object's centre and only centring y left the thumb 7.4cm away
        # along x -- right off the end of a 7cm can -- so the thumb never opposed and the
        # four fingers simply pressed the can down onto the table. Lifting then removed
        # the table's reaction and the grip vanished with it.
        jaw = 0.5 * (offset + thumb_offset)
        # Raise the grip until the lowest fingertip clears the table. Centring the jaw
        # on the can's own centre put the fingertips at z 0.398 against a table top of
        # 0.400 -- the hand closed into the table, not around the can.
        # Along the jaw line, sit where the fingers wrap the object rather than at the
        # exact midpoint: the two jaws are 12.9cm apart but the can is only 7.1cm across,
        # so a midpoint placement leaves both of them 2.9cm short of the surface and the
        # fingers close on air. Biasing toward the fingers puts them on the can and lets
        # the closing sweep carry the thumb onto the far side.
        jaw_line = thumb_offset - offset
        grasp = (
            bottle
            - jaw
            - JAW_BIAS_TOWARD_FINGERS * jaw_line
            + np.array([0.0, 0.0, GRASP_HEIGHT_BIAS])
            + GRASP_POSITION_CORRECTION
        )
        clearance = 0.5 * (float(np.linalg.norm(thumb_offset - offset)) - width)
        if clearance < GRASP_CLEARANCE:
            raise RuntimeError(
                f"jaw too narrow: {np.linalg.norm(thumb_offset - offset)*100:.1f}cm aperture "
                f"around a {width*100:.1f}cm object leaves {clearance*1000:.1f}mm per side"
            )
        # Both jaws must land on the object's body, not one on it and one past its end.
        if abs(thumb_offset[2] - offset[2]) > height - 2.0 * GRASP_CLEARANCE:
            raise RuntimeError(
                f"jaws are {abs(thumb_offset[2]-offset[2])*100:.1f}cm apart in height, too far "
                f"for a {height*100:.1f}cm object: lower GRASP_TILT_DEGREES to bring them level"
            )
        pregrasp = grasp - axis * APPROACH_STANDOFF

        basket = np.asarray(self.data.geom_xpos[self.model.geom("place_basket_bottom").id])
        drop = basket + np.array([0.0, 0.0, 0.5 * height + 0.06])
        place = drop - jaw
        return {
            # The standoff is already well clear of the bottle, so "ready" is just the
            # standoff itself. Lifting it another 12cm on top of a 20cm retreat put it
            # outside the arm's reach at every tilt.
            # Reached from above: a straight joint-space move from the attention stance
            # to the standoff dragged the forearm and the whole hand through the table
            # (measured 65mm of penetration). Coming down onto it keeps the path clear.
            "hover": grasp + np.array([0.0, 0.0, HOVER_HEIGHT]),
            "ready": pregrasp,
            "pregrasp": pregrasp,
            "grasp": grasp,
            "lift": grasp + np.array([0.0, 0.0, LIFT_HEIGHT]),
            "transfer": place + np.array([0.0, 0.0, LIFT_HEIGHT]),
            "lower": place,
        }

    def _solve_poses(self) -> dict[str, dict[str, np.ndarray]]:
        # Each phase is seeded from the previous solution rather than from RIGHT_SEED,
        # so the phases stay on one continuous IK branch (the same continuous-seeding
        # trick correlllab/rh56_controller uses for its waypoint chains). Re-seeding
        # every phase from scratch both fails to converge on the far poses and risks
        # branch jumps between neighbouring waypoints.
        #
        # The grasp itself is the most constrained pose, so it is solved first, from the
        # neutral seed; the approach is then walked backwards off it in small steps.
        # Solving the standoff first and seeding the grasp from it fails: the standoff's
        # own branch is a poor seed for the grasp and the grasp then does not converge.
        centers = self._phase_centers()
        poses: dict[str, np.ndarray] = {}
        poses["grasp"] = solve_pose_ik(
            self.model, "right", centers["grasp"], SIDE_GRASP_ORIENTATION, RIGHT_SEED
        )

        # Walk backwards off the grasp in small steps: a single jump to the standoff
        # leaves the reachable set even though both endpoints are reachable.
        seed = poses["grasp"]
        for start_phase, end_phase, steps in (("grasp", "pregrasp", 6),):
            start, end = centers[start_phase], centers[end_phase]
            seed = poses[start_phase]
            for step in range(1, steps + 1):
                seed = solve_pose_ik(
                    self.model, "right", start + (end - start) * (step / steps), SIDE_GRASP_ORIENTATION, seed
                )
            poses[end_phase] = seed
        poses["hover"] = poses["pregrasp"]
        poses["ready"] = poses["pregrasp"]

        # The carry waypoints only need to clear the table, so how high they ride is
        # negotiable -- unlike the grasp, which is fixed by where the object is. Ask for
        # the full lift and settle for less rather than aborting the whole trial: at the
        # release side a 10cm hover has no IK solution even though the release pose
        # itself does, and that alone used to fail the run before the robot moved.
        seed = poses["grasp"]
        for phase in ("lift", "transfer", "lower"):
            target = centers[phase]
            hover = LIFT_HEIGHT if phase in ("lift", "transfer") else 0.0
            for reduced in (hover, 0.75 * hover, 0.5 * hover, 0.25 * hover):
                candidate = target - np.array([0.0, 0.0, hover - reduced])
                try:
                    seed = solve_pose_ik(
                        self.model, "right", candidate, SIDE_GRASP_ORIENTATION, seed
                    )
                except RuntimeError:
                    continue
                poses[phase] = seed
                break
            else:
                raise RuntimeError(f"no reachable {phase} pose near {target.tolist()}")
        return {"right": poses}

    def step_to(
        self,
        targets: dict[str, np.ndarray],
        seconds: float,
        viewer=None,
        stop_on_grasp: bool = False,
    ) -> bool:
        return self.step_path({group: [target] for group, target in targets.items()}, [seconds], viewer, stop_on_grasp=stop_on_grasp)

    def step_path(
        self,
        waypoints: dict[str, list[np.ndarray]],
        durations: list[float],
        viewer=None,
        stop_on_grasp: bool = False,
    ) -> bool:
        # A single ease-in/ease-out envelope spans the *whole* multi-waypoint path, with plain
        # linear blending between the intermediate poses in between. Unlike calling step_to()
        # once per waypoint, this never decelerates to a stop at an intermediate pose -- only at
        # the true start and end of the merged motion -- which is what keeps a multi-phase reach
        # (approach -> descend -> close, or lift -> carry -> lower) reading as one continuous
        # human-like motion instead of a series of stop-motion segments.
        groups = list(waypoints)
        paths = {group: [self.data.qpos[self._qpos_for(group)].copy(), *waypoints[group]] for group in groups}
        dofs = {group: self._dofs_for(group) for group in groups}
        cumulative = np.cumsum([0.0, *durations])
        total_duration = cumulative[-1]
        cumulative_fraction = cumulative / total_duration
        steps = max(1, int(total_duration / self.model.opt.timestep))
        for index in range(steps):
            eased = quintic((index + 1) / steps)
            segment = min(int(np.searchsorted(cumulative_fraction, eased, side="right")) - 1, len(durations) - 1)
            segment = max(segment, 0)
            span = cumulative_fraction[segment + 1] - cumulative_fraction[segment]
            local_t = 0.0 if span <= 0 else (eased - cumulative_fraction[segment]) / span
            for group in groups:
                start, end = paths[group][segment], paths[group][segment + 1]
                position = start + local_t * (end - start)
                self.data.ctrl[self._ctrl_for(group)] = position
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            offenders = self._table_contacts()
            if offenders:
                detail = ", ".join(f"{body} {depth:.1f}mm" for body, depth in sorted(offenders.items()))
                raise RuntimeError(f"trajectory aborted: {detail} inside the table")
            if stop_on_grasp:
                groups = self._right_contact_groups()
                required = {"thumb", "fingers"} if self.REQUIRE_THUMB_OPPOSITION else {"fingers"}
                if required <= groups:
                    return True
            if viewer:
                viewer.sync()
                time.sleep(self.model.opt.timestep)
        return not stop_on_grasp

    def _qpos_for(self, group: str) -> np.ndarray:
        side, kind = group.split("_")
        return self.arm_qpos[side] if kind == "arm" else self.hand_qpos[side]

    def _ctrl_for(self, group: str) -> np.ndarray:
        side, kind = group.split("_")
        return self.arm_actuators[side] if kind == "arm" else self.hand_actuators[side]

    def _dofs_for(self, group: str) -> np.ndarray:
        side, kind = group.split("_")
        return self.arm_dofs[side] if kind == "arm" else self.hand_dofs[side]

    def _unsafe_table_contact(self) -> bool:
        return bool(self._table_contacts())

    def _table_contacts(self) -> dict[str, float]:
        """Which arm/hand parts are inside the table, and how deep, in mm.

        Named rather than boolean: "arm or hand contacted the table" gave no clue which
        part or which phase, and every diagnosis of it needed a separate script.
        """
        table = self.model.geom("table_top").id
        offenders: dict[str, float] = {}
        for contact in self.data.contact[: self.data.ncon]:
            if table not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == table else contact.geom1
            name = self.model.geom(other).name or ""
            if not (name.startswith(("openarm_left_link", "openarm_right_link")) or self._hand_side(other) is not None):
                continue
            body = self.model.body(int(self.model.geom_bodyid[other])).name or name
            if float(contact.dist) > -TABLE_CONTACT_TOLERANCE:
                continue
            offenders[body] = min(offenders.get(body, 0.0), float(contact.dist) * 1000.0)
        return offenders

    def _hand_side(self, geom: int) -> str | None:
        body = int(self.model.geom_bodyid[geom])
        while body:
            name = self.model.body(body).name or ""
            for side in ("left", "right"):
                if name.startswith(f"{HAND_PREFIX}{side}_"):
                    return side
            body = int(self.model.body_parentid[body])
        return None

    def _bottle_hand_contacts(self, side: str) -> int:
        bottle = self.model.geom("pick_bottle_collision").id
        count = 0
        for contact in self.data.contact[: self.data.ncon]:
            if bottle not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == bottle else contact.geom1
            count += self._hand_side(other) == side
        return count

    def preshape_hand(self, side: str) -> None:
        """Open the four fingers and swing the thumb into opposition.

        The thumb is positioned *before* the arm travels, never during the close:
        that is the ordering the reference hand uses (rh56_controller
        grasp_executor._run_thumb_reflex), and it is what lets the hand arrive
        already straddling the object -- an opposed thumb widens the jaw from
        3.5cm to 9.6cm, so the 6cm bottle passes between the jaws untouched.
        """
        for name, actuator in self.finger_actuator[side].items():
            self.data.ctrl[actuator] = self.open_ctrl[side][name]
        yaw_actuator, _, opposed = self.thumb_yaw[side]
        self.data.ctrl[yaw_actuator] = opposed

    def close_until_contact(
        self,
        side: str,
        fingers: tuple[str, ...],
        viewer=None,
        force_target: float = CONTACT_FORCE_TARGET_N,
    ) -> dict[str, float]:
        """Close `fingers` a step at a time, holding each one once it presses.

        This is the sim counterpart of the hardware adaptive force phase: the
        fingers are driven purely through `data.ctrl`, never by writing qpos, so
        MuJoCo's contact solver is what actually stops them -- that is what keeps
        them on the bottle's surface instead of passing through it. A finger whose
        measured normal force reaches `force_target` stops being commanded further
        while the others keep closing.
        """
        actuators = {name: self.finger_actuator[side][name] for name in fingers}
        closed = {name: self.closed_ctrl[side][name] for name in fingers}
        held: dict[str, float] = {}
        steps_per_iteration = max(1, int(CLOSE_SETTLE_SECONDS / self.model.opt.timestep))

        for _ in range(CLOSE_MAX_ITERATIONS):
            forces = self._finger_contact_forces(side)
            for name, actuator in actuators.items():
                if name in held:
                    continue
                if forces[name] >= force_target:
                    held[name] = float(self.data.ctrl[actuator])
                    continue
                lower, upper = self.model.actuator_ctrlrange[actuator]
                step = CLOSE_STEP_FRACTION * (upper - lower)
                target = closed[name]
                current = float(self.data.ctrl[actuator])
                delta = float(np.clip(target - current, -step, step))
                self.data.ctrl[actuator] = np.clip(current + delta, lower, upper)
            if len(held) == len(actuators):
                break
            self._advance(steps_per_iteration, viewer)

        # Let the contacts settle at the final commanded pressure.
        self._advance(steps_per_iteration * 4, viewer)
        return self._finger_contact_forces(side)

    def _advance(self, steps: int, viewer=None) -> None:
        """Step physics while holding the arm on its commanded pose."""
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if viewer:
                viewer.sync()
                time.sleep(self.model.opt.timestep)

    def _finger_contact_forces(self, side: str) -> dict[str, float]:
        """Per-finger normal contact force (N) against the bottle.

        Mirrors MujocoBridge.get_contacts() in correlllab/rh56_controller: read the
        real contact normal out of mj_contactForce rather than just counting contact
        points, so closing can stop on measured force the way the hardware's adaptive
        force phase does.
        """
        bottle = self.model.geom("pick_bottle_collision").id
        forces: dict[str, float] = {name: 0.0 for name in FINGER_NAMES}
        wrench = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if bottle not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == bottle else contact.geom1
            finger = self._finger_of(other, side)
            if finger is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            forces[finger] += abs(float(wrench[0]))
        return forces

    def _finger_of(self, geom: int, side: str) -> str | None:
        """Which finger (thumb/index/middle/ring/pinky) a geom belongs to."""
        body = int(self.model.geom_bodyid[geom])
        prefix = f"{HAND_PREFIX}{side}_"
        while body:
            name = self.model.body(body).name or ""
            if name.startswith(prefix):
                tail = name[len(prefix) :]
                for finger in FINGER_NAMES:
                    if tail.startswith(finger):
                        return finger
                return None
            body = int(self.model.body_parentid[body])
        return None

    def _right_contact_groups(self) -> set[str]:
        """Which digit group(s) of the right hand currently touch the bottle.

        Treats the hand like a two-jaw gripper: the thumb is one jaw, the other
        four fingers are the other. A secure grasp needs contact from both --
        that's an opposed pinch, not just fingers piled on one side.
        """
        bottle = self.model.geom("pick_bottle_collision").id
        groups: set[str] = set()
        for contact in self.data.contact[: self.data.ncon]:
            if bottle not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == bottle else contact.geom1
            body = int(self.model.geom_bodyid[other])
            while body:
                name = self.model.body(body).name or ""
                if name.startswith(f"{HAND_PREFIX}right_"):
                    groups.add("thumb" if "thumb" in name else "fingers")
                    break
                body = int(self.model.body_parentid[body])
        return groups

    def _right_grasp_is_secure(self) -> bool:
        """A grasp counts only when fingers are actually pressing on the bottle.

        Force-based, not position-based: the object is never lifted on the strength
        of "the fingers were commanded closed". At least two fingers must register a
        real normal force, and with REQUIRE_THUMB_OPPOSITION the thumb must be one
        of them (an opposed pinch, like the hardware's antipodal thumb-vs-fingers
        load distribution).
        """
        forces = self._finger_contact_forces("right")
        pressing = [name for name, force in forces.items() if force >= CONTACT_FORCE_TARGET_N]
        if self.REQUIRE_THUMB_OPPOSITION:
            return "thumb" in pressing and any(name != "thumb" for name in pressing)
        return len(pressing) >= 2

    def _proof_lift(self, viewer=None) -> tuple[float, float]:
        """Lift a few cm under physics and report how far the bottle came along.

        Nothing is welded or pinned during this: if the fingers are not really holding
        the bottle it simply stays on the table, which is exactly what we want to catch
        before committing to the carry.
        """
        before = float(self.data.qpos[self.bottle_qpos + 2])
        start = self.data.ctrl[self.arm_actuators["right"]].copy()
        target = solve_pose_ik(
            self.model,
            "right",
            self.data.site_xpos[self.ee_site_id["right"]] + np.array([0.0, 0.0, PROOF_LIFT_HEIGHT]),
            SIDE_GRASP_ORIENTATION,
            start,
        )
        steps = max(1, int(0.8 / self.model.opt.timestep))
        for index in range(steps):
            pose = start + (target - start) * ((index + 1) / steps)
            self.data.ctrl[self.arm_actuators["right"]] = pose
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if viewer:
                viewer.sync()
                time.sleep(self.model.opt.timestep)
        quaternion = self.data.qpos[self.bottle_qpos + 3 : self.bottle_qpos + 7]
        tilt = np.degrees(2 * np.arccos(np.clip(abs(float(quaternion[0])), 0.0, 1.0)))
        return float(self.data.qpos[self.bottle_qpos + 2]) - before, float(tilt)

    def run(self, viewer=None) -> None:
        self.poses = self._solve_poses()
        print("1/5 READY: attention stance (arms straight, fists closed), then raising the right arm")
        self.step_to({}, 1.5, viewer)
        # Up and over, never across: a direct joint-space move from the arms-down stance
        # to the standoff drags the hand through the table top. Rising to the hover pose
        # first keeps the whole path above it.
        self.step_to({"right_arm": self.poses["right"]["hover"], "right_hand": self.closed_hand["right"]}, 1.4, viewer)
        self.step_to({"right_arm": self.poses["right"]["ready"]}, 1.0, viewer)

        # Sequencing after correlllab/rh56_controller (grasp_executor._run_thumb_reflex):
        # the hand opens wide *before* travelling, the arm then slides in horizontally
        # so the bottle enters between the jaws -- nothing sweeps through it -- and the
        # fingers only close once the arm has arrived.
        print("2/5 REACH: opening the hand wide, arm slides in horizontally around the bottle")
        self.preshape_hand("right")
        self._advance(int(0.3 / self.model.opt.timestep), viewer)

        self.step_to({"right_arm": self.poses["right"]["pregrasp"]}, 1.2, viewer)
        self._advance(int(0.2 / self.model.opt.timestep), viewer)
        # The standoff must clear the bottle completely, or the "approach" starts from
        # inside it and the first contact is a shove rather than a grasp. Checked rather
        # than assumed: getting this wrong is silent until the bottle is already flying.
        touching = {name: force for name, force in self._finger_contact_forces("right").items() if force > 0.05}
        if touching:
            raise RuntimeError(
                f"standoff pose is already touching the bottle ({touching}); "
                f"increase APPROACH_STANDOFF (currently {APPROACH_STANDOFF:.2f}m)"
            )
        self.step_to({"right_arm": self.poses["right"]["grasp"]}, 1.0, viewer)

        print("3/5 GRASP: squeezing the four fingers onto the pre-opposed thumb")
        # The thumb is NOT closed here. It was positioned during the pre-shape and now
        # acts as the fixed jaw; the four fingers press the bottle against it. Flexing
        # the thumb as well made it reach the bottle first and swat it aside -- measured
        # 966 steps of thumb contact against 99 for the index finger, and the bottle
        # ended up on its side every time.
        forces = self.close_until_contact("right", ("index", "middle", "ring", "pinky"), viewer)
        print("     contact force per finger (N): " + ", ".join(f"{k}={v:.2f}" for k, v in forces.items()))

        # Prove the grasp by actually lifting a little and watching whether the bottle
        # comes with the hand. This is the only honest test: contact forces alone do not
        # distinguish "holding it" from "leaning on it".
        if not self._right_grasp_is_secure():
            raise RuntimeError(f"opposed grasp missing: {forces}")
        rise, tilt = self._proof_lift(viewer)
        print(f"     proof lift: bottle rose {rise*100:+.1f}cm, tilted {tilt:.0f} degrees")
        if rise < PROOF_LIFT_MIN_RISE or tilt > 15.0:
            raise RuntimeError(
                f"grasp failed: bottle did not come with the hand (rose {rise*100:.1f}cm, "
                f"needed {PROOF_LIFT_MIN_RISE*100:.1f}cm); forces {forces}"
            )

        print("4/5 CARRY: lifting and moving the YCB mustard bottle from A to B")
        carry_waypoints = {
            "right_arm": [self.poses["right"]["lift"], self.poses["right"]["transfer"], self.poses["right"]["lower"]],
        }
        self.step_path(carry_waypoints, [1.2, 1.5, 1.2], viewer)

        print("5/5 RELEASE: opening the right hand into the basket and retreating")
        self.step_to({"right_arm": self.poses["right"]["lower"], "right_hand": self.open_hand["right"]}, 0.8, viewer)
        retreat_waypoints = {
            "right_arm": [self.poses["right"]["transfer"], self.poses["right"]["pregrasp"]],
        }
        self.step_path(retreat_waypoints, [1.0, 1.0], viewer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scripted OpenArm right-hand five-finger pick-and-place demo")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--report", type=Path, default=Path("artifacts/physics_trials.json"))
    args = parser.parse_args()
    demo = Demo()
    if args.headless:
        if args.trials < 1:
            parser.error("--trials must be positive")
        results = []
        for _ in range(args.trials):
            demo = Demo()
            failure = None
            try:
                demo.run()
            except RuntimeError as error:
                failure = str(error)
            result = TrialResult(
                failure is None, failure,
                demo.data.qpos[demo.bottle_qpos:demo.bottle_qpos + 3].tolist(),
                float(demo.data.time),
            )
            results.append(asdict(result))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
        passed = sum(result["success"] for result in results)
        print(f"Physics trials: {passed}/{len(results)} passed; report: {args.report}")
        if passed != len(results):
            raise SystemExit(1)
        return
    with mujoco.viewer.launch_passive(demo.model, demo.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = demo.model.camera("overhead").id
        try:
            demo.run(viewer)
        except RuntimeError as failure:
            # In the viewer the point is to *watch* what the robot does, so a failed
            # grasp leaves the scene up for inspection instead of tearing it down.
            print(f"\n!! sequence stopped: {failure}")
            print("   the scene is left as-is -- rotate the view to inspect the hand")
        while viewer.is_running():
            mujoco.mj_step(demo.model, demo.data)
            viewer.sync()


if __name__ == "__main__":
    main()
