"""The simulated scene: model, data, index tables and geometry queries.

Scene owns the MjModel/MjData and knows *where things are*: arm and hand indices,
the object and basket geometry, per-finger contact forces. It never plans and never
steps time; planner.py and executor.py do that on top of it.
"""

from __future__ import annotations

import mujoco
import numpy as np

from simulation.five_finger_model import BASKET_POSITION_B, HAND_PREFIX, PICK_POSITION_A, build_five_finger_model
from simulation.pick_place.config import (
    ARM_ACTUATORS,
    ARM_JOINTS,
    ATTENTION_RIGHT,
    BASKET_CONTACT_TOLERANCE,
    BOTTLE_JOINT,
    CARRY_CLEARANCE_ABOVE_RIM,
    EE_SITE,
    FINGER_NAMES,
    OBJECT_GEOM,
    TABLE_CONTACT_TOLERANCE,
)
from simulation.pick_place.kinematics import hand_pose, natural_grasp_frame


class Scene:
    def __init__(self, pick_position=PICK_POSITION_A, basket_position=BASKET_POSITION_B) -> None:
        self.pick_position = tuple(float(v) for v in pick_position)
        self.basket_position = tuple(float(v) for v in basket_position)
        self.model = build_five_finger_model(
            pick_bottle=True, pick_position=self.pick_position, basket_position=self.basket_position
        )
        self.data = mujoco.MjData(self.model)
        # The grasp orientation is whatever the hand has when the wrist is straight in
        # the reference posture -- a natural, in-line hand, not a hand-tuned rotation.
        _, self.grasp_orientation = natural_grasp_frame(self.model)
        self._index_arms_and_hands()
        self._index_fingers()

        self.bottle_qpos = int(self.model.joint(BOTTLE_JOINT).qposadr[0])
        self.bottle_dof = int(self.model.joint(BOTTLE_JOINT).dofadr[0])
        self.bottle_body = self.model.body("pick_bottle").id
        self.object_geom = self.model.geom(OBJECT_GEOM).id
        self.table_geom = self.model.geom("table_top").id
        self.basket_geoms = {
            self.model.geom(f"place_basket_{name}").id for name in ("bottom", "left", "right", "front", "back")
        }
        self.ee_site_id = {side: self.model.site(EE_SITE[side]).id for side in ("left", "right")}
        self.palm_body = {side: self.model.body(f"{HAND_PREFIX}{side}_base").id for side in ("left", "right")}

        self.attention_pose = {"right": ATTENTION_RIGHT.copy(), "left": ATTENTION_RIGHT * np.array([-1, -1, -1, 1, -1, -1, -1])}
        self.reset()

    # ------------------------------------------------------------------ indexing
    def _index_arms_and_hands(self) -> None:
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

    def _index_fingers(self) -> None:
        """Per-finger actuator ids and open/closed ctrl values, so the adaptive
        closing loop can command and hold one finger at a time."""
        self.finger_actuator: dict[str, dict[str, int]] = {}
        self.open_ctrl: dict[str, dict[str, float]] = {}
        self.closed_ctrl: dict[str, dict[str, float]] = {}
        self.thumb_yaw: dict[str, tuple[int, float, float]] = {}
        for side in ("left", "right"):
            self.finger_actuator[side], self.open_ctrl[side], self.closed_ctrl[side] = {}, {}, {}
            for position, actuator in enumerate(self.hand_actuators[side]):
                name = self.model.actuator(int(actuator)).name or ""
                tail = name[len(f"{HAND_PREFIX}{side}_") :]
                lower, upper = self.model.actuator_ctrlrange[int(actuator)]
                if tail.startswith("thumb_yaw"):
                    # Opposition (abduction), not a closing DOF: set once as a pre-shape
                    # before the approach (rh56_controller's thumb reflex ordering).
                    self.thumb_yaw[side] = (int(actuator), float(lower), float(upper))
                    continue
                if tail.startswith("thumb_proximal"):
                    # Flexion: the thumb's closing DOF; equalities carry it to the other knuckles.
                    self.finger_actuator[side]["thumb"] = int(actuator)
                    self.open_ctrl[side]["thumb"] = float(lower)
                    self.closed_ctrl[side]["thumb"] = float(upper)
                    continue
                if tail.startswith("thumb"):
                    continue
                self.finger_actuator[side][tail] = int(actuator)
                self.open_ctrl[side][tail] = float(self.open_hand[side][position])
                # Closing drives to full curl and lets contact force stop the finger.
                self.closed_ctrl[side][tail] = float(upper)

    def reset(self) -> None:
        """Pose the scene at the start of an episode: attention stance, fists closed."""
        home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        # The object goes back to A upright and at rest whatever the keyframe holds.
        self.data.qpos[self.bottle_qpos : self.bottle_qpos + 7] = [*self.model.body("pick_bottle").pos, 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.bottle_dof : self.bottle_dof + 6] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        for side in ("left", "right"):
            self.data.qpos[self.arm_qpos[side]] = self.attention_pose[side]
            self.data.ctrl[self.arm_actuators[side]] = self.attention_pose[side]
            self.data.qpos[self.hand_qpos[side]] = self.closed_hand[side]
            self.data.ctrl[self.hand_actuators[side]] = self.closed_hand[side]
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ groups
    def qpos_for(self, group: str) -> np.ndarray:
        side, kind = group.split("_")
        return self.arm_qpos[side] if kind == "arm" else self.hand_qpos[side]

    def ctrl_for(self, group: str) -> np.ndarray:
        side, kind = group.split("_")
        return self.arm_actuators[side] if kind == "arm" else self.hand_actuators[side]

    # ------------------------------------------------------------------ object / basket
    def object_extents(self) -> tuple[float, float]:
        """(width across the grasp, full height) of the object's collision geom.
        MuJoCo packs geom_size per type: cylinder (radius, half-height), box (3 half-extents)."""
        geom = self.model.geom(OBJECT_GEOM)
        size = np.asarray(geom.size)
        if geom.type[0] in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            return 2.0 * float(size[0]), 2.0 * float(size[1])
        if geom.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE:
            return 2.0 * float(size[0]), 2.0 * float(size[0])
        return 2.0 * float(size[1]), 2.0 * float(size[2])

    def object_position(self) -> np.ndarray:
        return self.data.qpos[self.bottle_qpos : self.bottle_qpos + 3].copy()

    def object_quaternion(self) -> np.ndarray:
        return self.data.qpos[self.bottle_qpos + 3 : self.bottle_qpos + 7].copy()

    def object_bottom_z(self) -> float:
        return float(self.object_position()[2]) - 0.5 * self.object_extents()[1]

    def basket_floor(self) -> np.ndarray:
        return np.asarray(self.data.geom_xpos[self.model.geom("place_basket_bottom").id]).copy()

    def basket_rim_z(self) -> float:
        wall = self.model.geom("place_basket_left")
        return float(self.data.geom_xpos[wall.id][2] + wall.size[2])

    def carry_bottom_z(self) -> float:
        """Lowest world z the object's bottom may reach while being carried."""
        return self.basket_rim_z() + CARRY_CLEARANCE_ABOVE_RIM

    def wrist_position(self, side: str) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id[side]].copy()

    def wrist_rotation(self, side: str) -> np.ndarray:
        return self.data.site_xmat[self.ee_site_id[side]].reshape(3, 3).copy()

    # ------------------------------------------------------------------ contacts
    def hand_side(self, geom: int) -> str | None:
        body = int(self.model.geom_bodyid[geom])
        while body:
            name = self.model.body(body).name or ""
            for side in ("left", "right"):
                if name.startswith(f"{HAND_PREFIX}{side}_"):
                    return side
            body = int(self.model.body_parentid[body])
        return None

    def finger_of(self, geom: int, side: str) -> str | None:
        """Which finger (thumb/index/middle/ring/pinky) a geom belongs to."""
        body = int(self.model.geom_bodyid[geom])
        prefix = f"{HAND_PREFIX}{side}_"
        while body:
            name = self.model.body(body).name or ""
            if name.startswith(prefix):
                tail = name[len(prefix) :]
                return next((finger for finger in FINGER_NAMES if tail.startswith(finger)), None)
            body = int(self.model.body_parentid[body])
        return None

    def _is_robot_geom(self, geom: int) -> bool:
        name = self.model.geom(geom).name or ""
        return name.startswith(("openarm_left_link", "openarm_right_link")) or self.hand_side(geom) is not None

    def _penetrations(self, target_geoms: set[int], tolerance: float, ignore: set[int] = frozenset()) -> dict[str, float]:
        """Robot bodies penetrating any of `target_geoms` deeper than `tolerance`, in mm."""
        offenders: dict[str, float] = {}
        for contact in self.data.contact[: self.data.ncon]:
            if contact.geom1 not in target_geoms and contact.geom2 not in target_geoms:
                continue
            other = contact.geom2 if contact.geom1 in target_geoms else contact.geom1
            if other in ignore or not self._is_robot_geom(other):
                continue
            if float(contact.dist) > -tolerance:
                continue
            body = self.model.body(int(self.model.geom_bodyid[other])).name or self.model.geom(other).name
            offenders[body] = min(offenders.get(body, 0.0), float(contact.dist) * 1000.0)
        return offenders

    def table_contacts(self) -> dict[str, float]:
        """Which arm/hand parts are inside the table, and how deep (mm)."""
        return self._penetrations({self.table_geom}, TABLE_CONTACT_TOLERANCE)

    def basket_contacts(self) -> dict[str, float]:
        """Which arm/hand parts are colliding with the basket, and how deep (mm)."""
        return self._penetrations(self.basket_geoms, BASKET_CONTACT_TOLERANCE, ignore={self.object_geom})

    def finger_contact_forces(self, side: str) -> dict[str, float]:
        """Per-finger normal contact force (N) against the object, from mj_contactForce
        (as MujocoBridge.get_contacts() in correlllab/rh56_controller does)."""
        forces = {name: 0.0 for name in FINGER_NAMES}
        wrench = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.object_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.object_geom else contact.geom1
            finger = self.finger_of(other, side)
            if finger is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            forces[finger] += abs(float(wrench[0]))
        return forces

    def contact_groups(self, side: str) -> set[str]:
        """{'thumb', 'fingers'} subsets currently touching the object: the hand as a two-jaw gripper."""
        groups: set[str] = set()
        for contact in self.data.contact[: self.data.ncon]:
            if self.object_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.object_geom else contact.geom1
            finger = self.finger_of(other, side)
            if finger is not None:
                groups.add("thumb" if finger == "thumb" else "fingers")
        return groups
