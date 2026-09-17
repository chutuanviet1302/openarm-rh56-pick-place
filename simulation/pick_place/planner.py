"""Grasp planning: from an object position to a chain of wrist targets and IK solutions.

Everything is derived: jaw offsets are measured on the hand by forward kinematics,
wrist targets follow from the object/basket geometry, and the joint solutions are
chained from the straight-wrist reference posture. The planner never steps physics.

Debugging entry points:
    planner.centers(yaw)      -> wrist target per phase (world frame)
    planner.plan()            -> Plan with joints per phase, place yaw, paths
    planner.describe(plan)    -> printable table of targets vs FK of the solutions
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from simulation.five_finger_model import HAND_PREFIX
from simulation.pick_place import config as C
from simulation.pick_place.kinematics import rotation_z, solve_pose_ik, wrist_frame
from simulation.pick_place.scene import Scene

PHASE_ORDER = ("hover", "ready", "pregrasp", "grasp", "lift", "transfer", "lower")


@dataclass
class Plan:
    """Joint-space solutions for the right arm, one per phase, plus the Cartesian
    waypoint paths for the carry. `place_yaw_deg` is the hand's turn about vertical
    on the release side."""

    joints: dict[str, np.ndarray]
    paths: dict[str, list[np.ndarray]] = field(default_factory=dict)
    centers: dict[str, np.ndarray] = field(default_factory=dict)
    place_yaw_deg: float = 0.0

    def __getitem__(self, phase: str) -> np.ndarray:
        return self.joints[phase]


class GraspPlanner:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.model = scene.model
        self.orientation = scene.grasp_orientation
        self._local_jaw: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ hand geometry
    def local_jaw_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean fingertip and thumb-tip offsets from the wrist, in the wrist's own frame,
        with the hand pre-shaped (thumb opposed, fingers at GRASP_CLOSURE_FRACTION).

        Measured once by forward kinematics on a throwaway MjData, so the targets follow
        from where this particular hand's jaws actually end up.
        """
        if self._local_jaw is not None:
            return self._local_jaw
        scene, model = self.scene, self.model
        data = mujoco.MjData(model)
        for name, actuator in scene.finger_actuator["right"].items():
            joint = model.actuator_trnid[actuator, 0]
            opened, closed = scene.open_ctrl["right"][name], scene.closed_ctrl["right"][name]
            data.qpos[model.jnt_qposadr[joint]] = opened + C.GRASP_CLOSURE_FRACTION * (closed - opened)
        yaw_actuator, _, opposed = scene.thumb_yaw["right"]
        data.qpos[model.jnt_qposadr[model.actuator_trnid[yaw_actuator, 0]]] = opposed
        # Apply the hand's joint equalities by hand (no physics step here).
        for equality in range(model.neq):
            if model.eq_type[equality] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            driven, source = int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality])
            if not (model.joint(driven).name or "").startswith(f"{HAND_PREFIX}right_"):
                continue
            source_value = data.qpos[model.jnt_qposadr[source]]
            value = sum(float(c) * source_value**power for power, c in enumerate(model.eq_data[equality, :5]))
            data.qpos[model.jnt_qposadr[driven]] = np.clip(value, *model.jnt_range[driven])
        mujoco.mj_forward(model, data)

        site = scene.ee_site_id["right"]
        wrist = data.site_xpos[site].copy()
        rotation = data.site_xmat[site].reshape(3, 3)
        tips = [data.site_xpos[model.site(f"{HAND_PREFIX}right_right_{f}_tip").id] for f in ("index", "middle", "ring", "pinky")]
        thumb = data.site_xpos[model.site(f"{HAND_PREFIX}right_right_thumb_tip").id]
        self._local_jaw = (rotation.T @ (np.mean(tips, axis=0) - wrist), rotation.T @ (thumb - wrist))
        return self._local_jaw

    def jaw_offsets(self, orientation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(fingertip offset, thumb-tip offset) from the wrist in world axes for a wrist orientation."""
        fingers, thumb = self.local_jaw_offsets()
        return orientation @ fingers, orientation @ thumb

    # ------------------------------------------------------------------ targets
    def centers(
        self, object_position: np.ndarray, place_yaw_deg: float = 0.0, held_offset: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Wrist targets per phase, derived from the object, the basket and the jaw.

        `place_yaw_deg` turns the hand about world z on the release side only; the
        grasp itself always uses the straight-wrist orientation. `held_offset` is the
        object's *measured* position relative to the wrist (world axes, at the grasp
        orientation) once it is in hand; when given, the set-down is planned from where
        the object really sits instead of from the nominal jaw centre.
        """
        scene = self.scene
        bottle = np.asarray(object_position, dtype=float)
        width, height = scene.object_extents()
        offset, thumb_offset = self.jaw_offsets(self.orientation)

        # Straddle the object with the jaw centred on it in all three axes.
        jaw = 0.5 * (offset + thumb_offset)
        jaw_line = thumb_offset - offset
        jaw_axis = jaw_line / np.linalg.norm(jaw_line)
        grasp = (
            bottle
            - jaw
            + C.JAW_AXIS_BIAS * jaw_axis
            - C.JAW_BIAS_TOWARD_FINGERS * jaw_line
            + np.array([0.0, 0.0, C.GRASP_HEIGHT_BIAS])
            + C.GRASP_POSITION_CORRECTION
        )
        aperture = float(np.linalg.norm(jaw_line))
        clearance = 0.5 * (aperture - width)
        if clearance < C.GRASP_CLEARANCE:
            raise RuntimeError(
                f"jaw too narrow: {aperture*100:.1f}cm aperture around a {width*100:.1f}cm object "
                f"leaves {clearance*1000:.1f}mm per side"
            )
        if abs(thumb_offset[2] - offset[2]) > height - 2.0 * C.GRASP_CLEARANCE:
            raise RuntimeError(
                f"jaws are {abs(thumb_offset[2]-offset[2])*100:.1f}cm apart in height, too far for a "
                f"{height*100:.1f}cm object"
            )
        # Standoff backs off along the fingers' own horizontal pointing direction, so the
        # approach slides the open jaw straight onto the object.
        approach = np.array([offset[0], offset[1], 0.0])
        approach /= np.linalg.norm(approach)
        pregrasp = grasp - C.APPROACH_STANDOFF * approach
        hover = pregrasp + np.array([0.0, 0.0, C.HOVER_HEIGHT])

        # Wrist rise that puts the object's bottom the required clearance over the rim.
        lift_height = scene.carry_bottom_z() + C.CARRY_CLEARANCE_MARGIN + 0.5 * height - bottle[2]
        lift = grasp + np.array([0.0, 0.0, lift_height])

        drop = scene.basket_floor() + np.array([0.0, 0.0, 0.5 * height + C.PLACE_DROP_HEIGHT])
        turn = rotation_z(place_yaw_deg)
        if held_offset is None:
            place = drop - turn @ jaw + C.JAW_AXIS_BIAS * (turn @ jaw_axis)
        else:
            place = drop - turn @ held_offset
        transfer = place.copy()
        transfer[2] = lift[2]  # level carry above the rim
        return {
            "hover": hover,
            "ready": hover,
            "pregrasp": pregrasp,
            "grasp": grasp,
            "lift": lift,
            "transfer": transfer,
            "lower": place,
        }

    # ------------------------------------------------------------------ IK chain
    def _walk(self, start: np.ndarray, end: np.ndarray, seed: np.ndarray, steps: int, orientation_at=None) -> list[np.ndarray]:
        """Solve IK along a straight line from `start` to `end` in `steps`, chaining seeds.
        A single jump can leave the reachable set even when both endpoints are reachable."""
        path = []
        for step in range(1, steps + 1):
            fraction = step / steps
            orientation = self.orientation if orientation_at is None else orientation_at(fraction)
            seed = solve_pose_ik(self.model, "right", start + (end - start) * fraction, orientation, seed)
            path.append(seed)
        return path

    def plan(self, object_position: np.ndarray) -> Plan:
        """Solve the whole waypoint chain. Raises RuntimeError with the phase that failed."""
        centers = self.centers(object_position)
        joints: dict[str, np.ndarray] = {}
        # The grasp is the most constrained pose: solve it first from the reference
        # posture, then walk backwards off it to the standoff and up to the hover.
        joints["grasp"] = solve_pose_ik(self.model, "right", centers["grasp"], self.orientation, C.RIGHT_SEED)
        joints["pregrasp"] = self._walk(centers["grasp"], centers["pregrasp"], joints["grasp"], 6)[-1]
        joints["hover"] = self._walk(centers["pregrasp"], centers["hover"], joints["pregrasp"], 6)[-1]
        joints["ready"] = joints["hover"]
        joints["lift"] = solve_pose_ik(self.model, "right", centers["lift"], self.orientation, joints["grasp"])

        plan = Plan(joints, {}, centers, 0.0)
        self.plan_place(plan, object_position)
        return plan

    def plan_place(self, plan: Plan, object_position: np.ndarray, held_offset: np.ndarray | None = None) -> Plan:
        """Solve transfer + set-down (in place, on `plan`) from the lift pose.

        The place side may turn the hand about the vertical: the can stays upright
        either way, and it lets the basket sit where the grasp orientation alone
        cannot reach. Every intermediate solution is kept as a carry waypoint.
        Called once at planning time with the nominal jaw, and again after the grasp
        with the measured `held_offset` so the object -- not the wrist -- lands on the
        basket centre.
        """
        joints = plan.joints
        failures = []
        for yaw in C.PLACE_YAW_CANDIDATES_DEG:
            centers_yaw = self.centers(object_position, yaw, held_offset)
            try:
                transfer_path = self._walk(
                    centers_yaw["lift"], centers_yaw["transfer"], joints["lift"], C.CARRY_PATH_STEPS,
                    orientation_at=lambda f, yaw=yaw: rotation_z(yaw * f) @ self.orientation,
                )
                lower_path = self._walk(
                    centers_yaw["transfer"], centers_yaw["lower"], transfer_path[-1], C.CARRY_PATH_STEPS,
                    orientation_at=lambda f, yaw=yaw: rotation_z(yaw) @ self.orientation,
                )
            except RuntimeError as error:
                failures.append(f"yaw {yaw:+.0f}: {error}")
                continue
            joints["transfer"], joints["lower"] = transfer_path[-1], lower_path[-1]
            plan.paths = {"transfer": transfer_path, "lower": lower_path}
            plan.centers = centers_yaw
            plan.place_yaw_deg = yaw
            return plan
        raise RuntimeError("no reachable transfer/set-down pose at any hand yaw:\n  " + "\n  ".join(failures))

    # ------------------------------------------------------------------ debugging
    def describe(self, plan: Plan) -> str:
        """Table of phase -> target wrist position, FK of the solution, position error, wrist pitch."""
        lines = [f"{'phase':<10}{'target (x y z)':<28}{'reached (x y z)':<28}{'err mm':>8}{'wrist bend':>13}"]
        for phase in PHASE_ORDER:
            target = plan.centers[phase]
            reached, _ = wrist_frame(self.model, "right", plan.joints[phase])
            error = np.linalg.norm(reached - target) * 1000
            pitch = np.degrees(np.hypot(*(plan.joints[phase][i] for i in C.WRIST_BEND_INDICES)))
            lines.append(
                f"{phase:<10}{np.array2string(target, precision=3):<28}{np.array2string(reached, precision=3):<28}"
                f"{error:>8.1f}{pitch:>12.1f}"
            )
        lines.append(f"place yaw: {plan.place_yaw_deg:+.0f} deg")
        return "\n".join(lines)
