"""The simulated scene: model, data, index tables and geometry queries.

Scene owns the MjModel/MjData and knows *where things are*: arm and hand indices,
the object and basket geometry, per-finger contact forces. It never plans and never
steps time; planner.py and executor.py do that on top of it.
"""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from simulation.five_finger_model import (
    Conveyor,
    BASKET_HALF_WIDTH,
    BASKET_POSITION_B,
    HAND_PREFIX,
    PICK_POSITION_A,
    build_five_finger_model,
)
from simulation.pick_place.config import (
    ARM_ACTUATORS,
    ARM_JOINTS,
    ATTENTION_RIGHT,
    ROBOT_BODY_CONTACT_TOLERANCE,
    BASKET_CONTACT_TOLERANCE,
    BOTTLE_JOINT,
    CARRY_CLEARANCE_ABOVE_RIM,
    EE_SITE,
    FINGER_NAMES,
    GRASP_CLEARANCE,
    GRASP_CLOSURE_FRACTION,
    OBJECT_GEOM,
    REST_THUMB_UNOPPOSED,
    TABLE_CONTACT_TOLERANCE,
)
from simulation.objects import Placement, collision_points, geometric_center, quat_to_matrix
from simulation.objects import body_name as object_body_name
from simulation.pick_place.grasp_library import GraspTarget
from simulation.pick_place.kinematics import hand_pose, natural_grasp_frame, upright_tilt_degrees, wrist_frame


class Scene:
    def __init__(
        self, pick_position=PICK_POSITION_A, basket_position=BASKET_POSITION_B, *,
        arm_half_separation: float | None = None, left_arm_mount_yaw_deg: float | None = None,
        right_arm_mount_yaw_deg: float | None = None, basket_stand_height: float = 0.0,
        basket_floor_tilt_deg: float = 0.0, work_platform_height: float = 0.0,
        attention_deg: dict[str, Sequence[float]] | None = None,
        pick_object: str = "can", pick_pose: str = "upright", pick_yaw_deg: float = 0.0,
        extra_objects: Sequence[Placement] = (),
        conveyor: Conveyor | None = None,
        basket_half_size: tuple[float, float] | None = None, basket_wall_height: float | None = None,
    ) -> None:
        """`attention_deg`: optional per-arm rest pose override, {side: 7 joint angles
        in degrees}; arms not named keep ATTENTION_RIGHT (mirrored for the left).
        `pick_object`/`pick_pose`/`pick_yaw_deg`/`extra_objects`: which objects are on
        the table and how they rest (simulation.objects)."""
        self.pick_position = tuple(float(v) for v in pick_position)
        self.basket_position = tuple(float(v) for v in basket_position)
        self.model = build_five_finger_model(
            pick_bottle=True, pick_position=self.pick_position, basket_position=self.basket_position,
            arm_half_separation=arm_half_separation, left_arm_mount_yaw_deg=left_arm_mount_yaw_deg,
            right_arm_mount_yaw_deg=right_arm_mount_yaw_deg, basket_stand_height=basket_stand_height,
            basket_floor_tilt_deg=basket_floor_tilt_deg, work_platform_height=work_platform_height,
            pick_object=pick_object, pick_pose=pick_pose, pick_yaw_deg=pick_yaw_deg, extra_objects=extra_objects,
            conveyor=conveyor, basket_half_size=basket_half_size, basket_wall_height=basket_wall_height,
        )
        self.basket_half = np.array(basket_half_size or (BASKET_HALF_WIDTH, BASKET_HALF_WIDTH), dtype=float)
        self.conveyor = conveyor
        self.pick_object = pick_object
        # Every object on the table by registry key -> its free joint. The pick target
        # keeps the legacy pick_bottle names.
        # Object instances by name (the registry key unless named, e.g. "can_2") ->
        # free joint, and -> registry key (what the grasp library, geometry and
        # perception look up).
        self.object_joints = {pick_object: BOTTLE_JOINT}
        self.object_joints.update({p.label: f"{object_body_name(p.label)}_joint" for p in extra_objects})
        self.object_types = {pick_object: pick_object}
        self.object_types.update({p.label: p.key for p in extra_objects})
        # Height of the surface the object is picked from and set down on.
        self.work_surface_z = float(work_platform_height)
        if conveyor is not None:
            # Objects stand on the belt; it is "floor" for every clearance/contact rule.
            self.work_surface_z = conveyor.top_z(self.work_surface_z)
        self.basket_stand_height = float(basket_stand_height)
        self.basket_floor_tilt_deg = float(basket_floor_tilt_deg)
        self.data = mujoco.MjData(self.model)
        # Grasp-library target of the current pick (Scene.set_grasp_target), and the
        # finger pre-shape the planner measures the jaw at. None / the configured
        # fraction keep the original can behaviour.
        self.grasp_target: GraspTarget | None = None
        self.grasp_closure_fraction = GRASP_CLOSURE_FRACTION
        # The grasp orientation is whatever the hand has when the wrist is straight in
        # the reference posture -- a natural, in-line hand, not a hand-tuned rotation.
        self.grasp_orientation = {
            side: natural_grasp_frame(self.model, side)[1] for side in ("left", "right")
        }
        self._index_arms_and_hands()  # also indexes the fingers (rest hand needs them)

        self.set_target(pick_object)
        self.table_geoms = {self.model.geom("table_top").id}
        # The work platform is table for every safety check (hand contact aborts).
        if work_platform_height > 0.0:
            self.table_geoms.add(self.model.geom("work_platform").id)
        if conveyor is not None:
            self.table_geoms.add(self.model.geom("conveyor_belt_top").id)
            self.conveyor_actuator = self.model.actuator("conveyor_drive").id
            self.conveyor_qpos = int(self.model.joint("conveyor_slide").qposadr[0])
        self.basket_geoms = {
            self.model.geom(f"place_basket_{name}").id for name in ("bottom", "left", "right", "front", "back")
        }
        # The stand under a raised basket is an obstacle exactly like the basket: the
        # planner's hand-contact checks and the executor's abort both cover it.
        if self.basket_stand_height > 0.0:
            self.basket_geoms.add(self.model.geom("place_basket_stand").id)
        # V-floor insert plates (five_finger_model, basket_floor_tilt_deg).
        self.basket_floor_geoms = {self.model.geom("place_basket_bottom").id}
        if self.basket_floor_tilt_deg:
            for name in ("a", "b"):
                geom = self.model.geom(f"place_basket_slope_{name}").id
                self.basket_geoms.add(geom)
                self.basket_floor_geoms.add(geom)
        # The robot's own support: pedestal and torso. Arm links 0/1 are bolted to
        # the torso and excluded from these checks (see robot_body_contacts).
        self.robot_body_geoms = {self.model.geom("robot_riser").id, self.model.geom("openarm_body_link0_collision").id}
        self.ee_site_id = {side: self.model.site(EE_SITE[side]).id for side in ("left", "right")}
        self.palm_body = {side: self.model.body(f"{HAND_PREFIX}{side}_base").id for side in ("left", "right")}

        self.attention_pose = {"right": ATTENTION_RIGHT.copy(), "left": ATTENTION_RIGHT * np.array([-1, -1, -1, 1, -1, -1, -1])}
        for side, degrees in (attention_deg or {}).items():
            pose = np.radians(np.asarray(degrees, dtype=float))
            if side not in self.attention_pose or pose.shape != (7,):
                raise ValueError("attention_deg must map 'left'/'right' to 7 joint angles")
            self.attention_pose[side] = pose
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
        self._index_fingers()
        # Resting hand: a fist, with the thumb swung out of opposition where
        # REST_THUMB_UNOPPOSED says so. The left thumb, opposed in the fist, lies
        # folded across the index finger and hooked on it when the hand opened
        # (flexion stuck at 0.38 with an open command), so the left hand missed its
        # first grasp and had to re-grasp (2026-09-24).
        self.rest_hand = {}
        for side in ("left", "right"):
            rest = self.closed_hand[side].copy()
            if REST_THUMB_UNOPPOSED[side]:
                yaw_actuator, unopposed, _ = self.thumb_yaw[side]
                rest[list(self.hand_actuators[side]).index(yaw_actuator)] = unopposed
            self.rest_hand[side] = rest

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
                    #
                    # The limit, even though the thumb's arc carries its tip *across* the
                    # opposing fingers: swept free of any object the jaw narrows to 58mm at
                    # flexion 0.40 and widens again to 63mm at the limit, and a thumb pressing
                    # an object past that peak does lose force as it closes (measured mid-carry,
                    # 16N -> 10N -> released). Stopping it at the peak is worse, not better:
                    # these are position servos, so the squeeze *is* the commanded overshoot
                    # into the object, and a thumb told to stop where it can already reach
                    # presses with nothing (tried: 17/20 -> 10/20, objects dropped on the lift).
                    # Closing is force-driven anyway -- close_until_contact stops each finger on
                    # its contact force, long before the limit.
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

    def jaw_offsets_at(self, side: str, thumb_flexion: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(four-finger tip centroid, thumb tip), in the wrist frame, with the hand
        pre-shaped for a grasp: thumb opposed, fingers part closed. `thumb_flexion`
        overrides the thumb's closing DOF; by default it takes the same pre-shape as
        the fingers. Kinematics only -- no physics step, no contact."""
        model = self.model
        data = mujoco.MjData(model)
        for name, actuator in self.finger_actuator[side].items():
            joint = model.actuator_trnid[actuator, 0]
            opened, closed = self.open_ctrl[side][name], self.closed_ctrl[side][name]
            value = opened + self.grasp_closure_fraction * (closed - opened)
            if name == "thumb" and thumb_flexion is not None:
                value = thumb_flexion
            data.qpos[model.jnt_qposadr[joint]] = value
        yaw_actuator, _, opposed = self.thumb_yaw[side]
        data.qpos[model.jnt_qposadr[model.actuator_trnid[yaw_actuator, 0]]] = opposed
        # Apply the hand's coupled-joint equalities by hand (no physics step here).
        for equality in range(model.neq):
            if model.eq_type[equality] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            driven, source = int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])
            if not (model.joint(driven).name or "").startswith(f"{HAND_PREFIX}{side}_"):
                continue
            source_value = data.qpos[model.jnt_qposadr[source]]
            value = sum(float(c) * source_value**power for power, c in enumerate(model.eq_data[equality, :5]))
            data.qpos[model.jnt_qposadr[driven]] = np.clip(value, *model.jnt_range[driven])
        mujoco.mj_forward(model, data)
        site = self.ee_site_id[side]
        wrist, rotation = data.site_xpos[site].copy(), data.site_xmat[site].reshape(3, 3)
        tips = [
            data.site_xpos[model.site(f"{HAND_PREFIX}{side}_{side}_{finger}_tip").id]
            for finger in ("index", "middle", "ring", "pinky")
        ]
        thumb = data.site_xpos[model.site(f"{HAND_PREFIX}{side}_{side}_thumb_tip").id]
        return rotation.T @ (np.mean(tips, axis=0) - wrist), rotation.T @ (thumb - wrist)

    def reset(self) -> None:
        """Pose the scene at the start of an episode: attention stance, fists closed."""
        home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        # Every object goes back to its spawn pose, at rest, whatever the keyframe holds.
        for joint_name in self.object_joints.values():
            joint = self.model.joint(joint_name)
            body = self.model.body(int(joint.bodyid[0]))
            qpos, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
            self.data.qpos[qpos : qpos + 7] = [*body.pos, *body.quat]
            self.data.qvel[dof : dof + 6] = 0.0
        self.data.time = 0.0
        if self.conveyor is not None:
            self.data.qpos[self.conveyor_qpos] = 0.0
            self.data.ctrl[self.conveyor_actuator] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for side in ("left", "right"):
            self.data.qpos[self.arm_qpos[side]] = self.attention_pose[side]
            self.data.ctrl[self.arm_actuators[side]] = self.attention_pose[side]
            self.data.qpos[self.hand_qpos[side]] = self.rest_hand[side]
            self.data.ctrl[self.hand_actuators[side]] = self.rest_hand[side]
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ groups
    def qpos_for(self, group: str) -> np.ndarray:
        side, kind = group.split("_")
        return self.arm_qpos[side] if kind == "arm" else self.hand_qpos[side]

    def ctrl_for(self, group: str) -> np.ndarray:
        side, kind = group.split("_")
        return self.arm_actuators[side] if kind == "arm" else self.hand_actuators[side]

    # ------------------------------------------------------------------ object / basket
    def set_target(self, key: str) -> None:
        """Make `key` the pick object: every object/contact/basket query and the
        planner follow it; the other objects on the table become obstacles."""
        joint = self.model.joint(self.object_joints[key])
        self.pick_object = key
        self.pick_type = self.object_types[key]
        self.bottle_qpos = int(joint.qposadr[0])
        self.bottle_dof = int(joint.dofadr[0])
        self.bottle_body = int(joint.bodyid[0])
        body_name = self.model.body(self.bottle_body).name
        self.object_geom = self.model.geom(OBJECT_GEOM if body_name == "pick_bottle" else f"{body_name}_collision").id
        self._center_offset = geometric_center(self.pick_type)
        self.grasp_target = None
        self.grasp_closure_fraction = GRASP_CLOSURE_FRACTION

    def other_object_geoms(self) -> set[int]:
        """Collision geoms of every object but the pick object (obstacles)."""
        geoms = set()
        for key, joint_name in self.object_joints.items():
            if key == self.pick_object:
                continue
            body = self.model.body(int(self.model.joint(joint_name).bodyid[0])).name
            geoms.add(self.model.geom(OBJECT_GEOM if body == "pick_bottle" else f"{body}_collision").id)
        return geoms

    def object_in_basket(self, key: str | None = None) -> bool:
        """Dropped in, i.e. not fallen out: the object's centre is over the box floor
        (inside the walls), it hangs no higher than a hand-width over the rim (not
        still held), and it touches neither the table, the platform nor the belt.
        Resting on other objects in the box, even sticking out above the rim, counts."""
        key = key or self.pick_object
        centre = self.object_position_of(key)
        offset = np.abs(centre[:2] - self.basket_floor()[:2])
        if np.any(offset > self.basket_half) or centre[2] > self.basket_rim_z() + 0.10:
            return False
        joint = self.model.joint(self.object_joints[key])
        body = int(joint.bodyid[0])
        for contact in self.data.contact[: self.data.ncon]:
            geoms = (contact.geom1, contact.geom2)
            if any(int(self.model.geom_bodyid[g]) == body for g in geoms) and any(g in self.table_geoms for g in geoms):
                return False
        return True

    def set_grasp_target(self, target: GraspTarget, side: str = "right") -> None:
        """Take the object as the grasp library says (grasp_library.select_grasp).

        Also derives the finger pre-shape the jaw is measured at: the configured
        GRASP_CLOSURE_FRACTION when that leaves GRASP_CLEARANCE per side around the
        object's width (the can: unchanged), otherwise the most-closed shape that
        does (a 75mm apple does not fit the 72mm jaw at 0.20)."""
        self.grasp_target = target
        self.fit_closure_fraction(target.entry.width, side)

    def fit_closure_fraction(self, width: float, side: str = "right") -> float:
        """Set (and return) the most-closed finger pre-shape, from GRASP_CLOSURE_FRACTION
        down, whose jaw leaves GRASP_CLEARANCE per side around `width`."""
        for fraction in np.arange(GRASP_CLOSURE_FRACTION, -1e-9, -0.01):
            self.grasp_closure_fraction = float(fraction)
            fingers, thumb = self.jaw_offsets_at(side)
            if 0.5 * (float(np.linalg.norm(thumb - fingers)) - width) >= GRASP_CLEARANCE:
                return self.grasp_closure_fraction
        raise RuntimeError(f"a {width*1000:.0f}mm object does not fit the open jaw")

    def object_tilt_deg(self) -> float:
        """Tilt of the pick object: angle between its axis and world up (upright can),
        or the grasp library's rule for the rest pose it was picked in (lying can/pear:
        axis off the horizontal; fruit: 0)."""
        if self.grasp_target is None:
            return upright_tilt_degrees(self.object_quaternion())
        return self.grasp_target.tilt_deg(self.object_pose()[:3, :3])

    def object_extents(self) -> tuple[float, float]:
        """(width across the grasp, full height) of the object: the grasp library's
        values when a target is set, else the collision geom's. MuJoCo packs
        geom_size per type: cylinder (radius, half-height), box (3 half-extents)."""
        if self.grasp_target is not None:
            return self.grasp_target.entry.width, self.grasp_target.entry.height
        geom = self.model.geom(OBJECT_GEOM)
        size = np.asarray(geom.size)
        if geom.type[0] in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            return 2.0 * float(size[0]), 2.0 * float(size[1])
        if geom.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE:
            return 2.0 * float(size[0]), 2.0 * float(size[0])
        return 2.0 * float(size[1]), 2.0 * float(size[2])

    def set_belt_speed(self, speed: float) -> None:
        """Belt speed (m/s along world y); the velocity servo holds it."""
        self.data.ctrl[self.conveyor_actuator] = float(speed)

    def object_position(self) -> np.ndarray:
        """The pick object's geometric centre (world). For the centred can this is its
        body origin, as it always was; a YCB scan's origin sits off its centre (the
        tuna can's by 34 mm), and every height rule (carry clearance, set-down) is
        about the centre."""
        origin = self.data.qpos[self.bottle_qpos : self.bottle_qpos + 3]
        if not self._center_offset.any():
            return origin.copy()
        return origin + quat_to_matrix(self.object_quaternion()) @ self._center_offset

    def object_position_of(self, name: str) -> np.ndarray:
        """Geometric centre (world) of any object instance on the table."""
        pose = self.object_pose(name)
        return pose[:3, :3] @ geometric_center(self.object_types[name]) + pose[:3, 3]

    def object_quaternion(self) -> np.ndarray:
        return self.data.qpos[self.bottle_qpos + 3 : self.bottle_qpos + 7].copy()

    def object_pose(self, key: str | None = None) -> np.ndarray:
        """Ground-truth 4x4 world pose of an object's body (= its OBJ frame); the pick
        target by default. Simulator state: for the `gt` backend and for scoring only,
        never inside a detector."""
        joint = self.model.joint(self.object_joints[key or self.pick_object])
        qpos = self.data.qpos[int(joint.qposadr[0]) : int(joint.qposadr[0]) + 7]
        pose = np.eye(4)
        pose[:3, :3] = quat_to_matrix(qpos[3:7])
        pose[:3, 3] = qpos[:3]
        return pose

    def object_bottom_z(self) -> float:
        return float(self.object_position()[2]) - 0.5 * self.object_extents()[1]

    def object_on_basket_floor(self) -> bool:
        """The object touches the basket's floor (the V insert plates count as floor)."""
        for contact in self.data.contact[: self.data.ncon]:
            pair = (contact.geom1, contact.geom2)
            if self.object_geom in pair and (pair[0] in self.basket_floor_geoms or pair[1] in self.basket_floor_geoms):
                return True
        return False

    def basket_floor(self) -> np.ndarray:
        return np.asarray(self.data.geom_xpos[self.model.geom("place_basket_bottom").id]).copy()

    def object_inside_basket(self, tolerance: float = 0.002) -> bool:
        """Containment success: the object's collision shape, at its current pose,
        lies inside the basket's inner walls (seen from above). For the upright can
        this is its radius all round, as before; a lying can resting against a wall
        is inside as long as its round side is (a fixed 50mm footprint called cans
        that had rolled to a wall "outside", 2026-09-28)."""
        pose = self.object_pose()
        points = collision_points(self.pick_type) @ pose[:3, :3].T + pose[:3, 3]
        offset = np.abs(points[:, :2] - self.basket_floor()[:2])
        return bool(np.all(offset <= self.basket_half - tolerance))

    def object_resting_in_basket(self, wall_contact: float = 0.002) -> bool:
        """The object lies on the basket floor with its whole footprint within the
        walls, touching them allowed (`wall_contact` of contact penetration). A lying
        can or pear rolls after the release and comes to rest against a wall: that
        is in the basket (grasp-library picks are judged by this)."""
        return self.object_on_basket_floor() and self.object_inside_basket(tolerance=-wall_contact)

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

    def hand_ctrl(self, side: str, open_fingers: tuple[str, ...] = (), thumb_opposed: bool = True) -> np.ndarray:
        """Hand ctrl vector: a fist with the named fingers uncurled; the thumb yaw stays
        opposed unless `thumb_opposed` is False."""
        ctrl = self.closed_hand[side].copy()
        actuators = list(self.hand_actuators[side])
        for name in open_fingers:
            ctrl[actuators.index(self.finger_actuator[side][name])] = self.open_ctrl[side][name]
        yaw_actuator, unopposed, opposed = self.thumb_yaw[side]
        ctrl[actuators.index(yaw_actuator)] = opposed if thumb_opposed else unopposed
        return ctrl

    def wrist_position_at(self, side: str, joints: np.ndarray) -> np.ndarray:
        """Wrist position for an arm joint vector, by FK on a throwaway MjData."""
        position, _ = wrist_frame(self.model, side, np.asarray(joints, dtype=float))
        return position

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

    def support_contacts(self) -> dict[str, float]:
        """Which arm/hand parts penetrate the floor, and how deep (mm)."""
        return self._penetrations(self.table_geoms, TABLE_CONTACT_TOLERANCE)

    def basket_contacts(self) -> dict[str, float]:
        """Which arm/hand parts are colliding with the basket, and how deep (mm)."""
        return self._penetrations(self.basket_geoms, BASKET_CONTACT_TOLERANCE, ignore={self.object_geom})

    def robot_body_contacts(self) -> dict[str, float]:
        """Arm/hand parts pressing into the robot's own pedestal or torso (mm). The
        shoulder links 0/1 are mounted on the torso and always touch it, so they
        are ignored; anything further out touching it is a collision."""
        mounts = {f"openarm_{side}_link{i}" for side in ("left", "right") for i in (0, 1)}
        return {
            body: depth
            for body, depth in self._penetrations(self.robot_body_geoms, ROBOT_BODY_CONTACT_TOLERANCE).items()
            if body not in mounts
        }

    def robot_side(self, geom: int) -> str | None:
        """'left'/'right' for any geom on that arm or hand, None for everything else
        (torso, table, basket, object)."""
        hand = self.hand_side(geom)
        if hand is not None:
            return hand
        body = int(self.model.geom_bodyid[geom])
        while body:
            name = self.model.body(body).name or ""
            for side in ("left", "right"):
                if name.startswith(f"openarm_{side}_link"):
                    return side
            body = int(self.model.body_parentid[body])
        return None

    def inter_arm_contacts(self, tolerance: float = 0.0) -> dict[str, float]:
        """Left-arm/hand bodies touching right-arm/hand bodies: {'a <-> b': depth mm}.
        The two arms must never touch -- neither the links nor the hands."""
        hits: dict[str, float] = {}
        for contact in self.data.contact[: self.data.ncon]:
            sides = {self.robot_side(contact.geom1), self.robot_side(contact.geom2)}
            if sides != {"left", "right"} or float(contact.dist) > -tolerance:
                continue
            names = sorted(self.model.body(int(self.model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2))
            key = f"{names[0]} <-> {names[1]}"
            hits[key] = min(hits.get(key, 0.0), float(contact.dist) * 1000.0)
        return hits

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

    def object_on_work_surface(self) -> bool:
        """The object rests on the table top or the work platform."""
        names = ["table_top"] + (["work_platform"] if self.work_surface_z > 0.0 else [])
        return any(self.object_touches(name) for name in names)

    def object_touches(self, geom_name: str) -> bool:
        """True while the object's collision geom is in contact with `geom_name`."""
        other = self.model.geom(geom_name).id
        for contact in self.data.contact[: self.data.ncon]:
            if {contact.geom1, contact.geom2} == {self.object_geom, other} and float(contact.dist) < 0.001:
                return True
        return False
