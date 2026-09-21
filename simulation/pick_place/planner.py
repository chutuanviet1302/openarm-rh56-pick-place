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

import copy

from dataclasses import dataclass, field

import mujoco
import numpy as np

from simulation.five_finger_model import HAND_PREFIX
from simulation.pick_place import config as C
from simulation.pick_place.kinematics import orientation_error, rotation_z, solve_pose_ik, wrist_frame
from simulation.pick_place.scene import Scene

PHASE_ORDER = ("hover", "ready", "pregrasp", "grasp", "lift", "transfer", "lower")


@dataclass
class Plan:
    """Joint-space solutions for one arm, one per phase, plus the Cartesian
    waypoint paths for the carry. `grasp_yaw_deg` is the hand's turn about vertical
    for the whole plan (the can is round, so the top grasp may come in from any
    heading the arm reaches); `place_yaw_deg` is the extra turn on the release side."""

    joints: dict[str, np.ndarray]
    paths: dict[str, list[np.ndarray]] = field(default_factory=dict)
    centers: dict[str, np.ndarray] = field(default_factory=dict)
    grasp_yaw_deg: float = 0.0
    place_yaw_deg: float = 0.0
    route_strategy: str = "direct"
    side: str = "right"

    def __getitem__(self, phase: str) -> np.ndarray:
        return self.joints[phase]


class GraspPlanner:
    def __init__(self, scene: Scene, side: str = "right") -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.scene = scene
        self.model = scene.model
        self.side = side
        # Copy of the model for clearance checks: the active hand's geoms carry a
        # collision margin so contacts are reported up to PATH_CLEARANCE away.
        self.check_model = copy.copy(scene.model)
        for geom in range(self.check_model.ngeom):
            if scene.hand_side(geom) == side:
                self.check_model.geom_margin[geom] = C.PATH_CLEARANCE
        self.base_orientation = scene.grasp_orientation[side]
        # Orientation of the current plan: base turned by the chosen grasp yaw.
        self.orientation = self.base_orientation
        self._local_jaw: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ hand geometry
    def local_jaw_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean fingertip and thumb-tip offsets from the wrist, in the wrist's own frame,
        with the hand pre-shaped (thumb opposed, fingers at GRASP_CLOSURE_FRACTION).

        Measured once by forward kinematics on a throwaway MjData, so the targets follow
        from where this particular hand's jaws actually end up.
        """
        if self._local_jaw is None:
            self._local_jaw = self.scene.jaw_offsets_at(self.side)
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
            - C.JAW_BIAS_TOWARD_FINGERS[self.side] * jaw_line
            + np.array([0.0, 0.0, C.GRASP_HEIGHT_BIAS])
            + C.GRASP_POSITION_CORRECTION * np.array([1.0, 1.0 if self.side == "right" else -1.0, 1.0])
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
        # Standoff backs off along the fingers' full 3-D pointing direction. For the
        # floor layout this is a top approach tilted toward the robot, not a horizontal
        # side-grasp.
        approach = offset.copy()
        approach /= np.linalg.norm(approach)
        pregrasp = grasp - C.APPROACH_STANDOFF * approach
        hover = pregrasp + np.array([0.0, 0.0, C.HOVER_HEIGHT])

        # Wrist rise that puts the object's bottom the required clearance over the rim.
        lift_height = scene.carry_bottom_z() + C.CARRY_CLEARANCE_MARGIN + 0.5 * height - bottle[2]
        lift = grasp + np.array([0.0, 0.0, lift_height])

        drop = (
            scene.basket_floor()
            + np.array([0.0, 0.0, 0.5 * height + C.PLACE_DROP_HEIGHT])
            + C.PLACE_POSITION_CORRECTION[self.side]
        )
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
    def joint_margin_degrees(self, joints: np.ndarray) -> float:
        ids = np.array([self.model.joint(name).id for name in C.ARM_JOINTS[self.side]])
        lower, upper = self.model.jnt_range[ids].T
        return float(np.degrees(np.min(np.minimum(joints - lower, upper - joints))))

    def _walk(self, start: np.ndarray, end: np.ndarray, seed: np.ndarray, steps: int, orientation_at=None) -> list[np.ndarray]:
        """Solve IK along a straight line from `start` to `end` in `steps`, chaining seeds.
        A single jump can leave the reachable set even when both endpoints are reachable."""
        path = []
        for step in range(1, steps + 1):
            fraction = step / steps
            orientation = self.orientation if orientation_at is None else orientation_at(fraction)
            seed = solve_pose_ik(self.model, self.side, start + (end - start) * fraction, orientation, seed)
            path.append(seed)
        return path

    def plan(self, object_position: np.ndarray, exclude_yaws_deg: tuple[float, ...] = ()) -> Plan:
        """Solve the whole waypoint chain. Raises RuntimeError with the phase that failed.

        Tries each grasp heading in C.GRASP_YAW_CANDIDATES_DEG (0 first, the reference
        posture) and keeps the first whose grasp, standoff, hover and lift all solve.
        `exclude_yaws_deg` skips headings already tried and found wanting in physics
        (the executor's contact / proof-lift checks), so a retry picks another.
        """
        failures = [f"grasp yaw {yaw:+.0f}: failed in physics, not retried" for yaw in exclude_yaws_deg]
        for yaw in C.GRASP_YAW_CANDIDATES_DEG:
            if yaw in exclude_yaws_deg:
                continue
            self.orientation = rotation_z(yaw) @ self.base_orientation
            try:
                plan = self._plan_pick(object_position)
            except RuntimeError as error:
                failures.append(f"grasp yaw {yaw:+.0f}: {error}")
                continue
            plan.grasp_yaw_deg = yaw
            self.plan_place(plan, object_position)
            return plan
        self.orientation = self.base_orientation
        raise RuntimeError("no reachable grasp at any hand yaw:\n  " + "\n  ".join(failures))

    def plan_pick(self, object_position: np.ndarray) -> Plan:
        """Plan only through proof-lift; used by the bimanual route preflight."""
        failures = []
        for yaw in C.GRASP_YAW_CANDIDATES_DEG:
            self.orientation = rotation_z(yaw) @ self.base_orientation
            try:
                plan = self._plan_pick(object_position)
                plan.grasp_yaw_deg = yaw
                return plan
            except RuntimeError as error:
                failures.append(f"yaw {yaw:+.0f}: {error}")
        raise RuntimeError("no reachable pick at any hand yaw:\n  " + "\n  ".join(failures))

    def find_raise(self, hover_joints: np.ndarray, hover_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Way point between the attention stance and `hover`: the plan's hand pose
        lifted above where the hand hangs at attention, so the arm rises first and
        only then travels over the table. The joint-space blends attention -> raise ->
        hover are the only unplanned motions; they are checked against the basket,
        the object (where it is *now*: at A before the pick, at B for the return) and
        the table with PATH_CLEARANCE of room, and the first clear height/offset wins.
        Returns (joints, centre); raises RuntimeError when nothing is clear."""
        hanging = self.scene.wrist_position_at(self.side, self.scene.attention_pose[self.side])
        attention = self.scene.attention_pose[self.side]
        obstacles = self.scene.basket_geoms | {self.scene.object_geom} | self.scene.table_geoms
        failures, feasible = [], []
        for extra in np.linspace(C.RAISE_ABOVE_HOVER, 0.0, 5):
            for dx, raw_dy in C.RAISE_XY_OFFSETS:
                dy = raw_dy if self.side == "right" else -raw_dy
                center = np.array([hanging[0] + dx, hanging[1] + dy, hover_center[2] + extra])
                label = f"+{extra*100:.0f}cm ({dx:+.2f},{dy:+.2f})"
                try:
                    candidate = solve_pose_ik(self.model, self.side, center, self.orientation, hover_joints)
                except RuntimeError as error:
                    failures.append(f"{label}: {error}")
                    continue
                margin = self.joint_margin_degrees(candidate)
                if margin < C.MIN_JOINT_MARGIN_DEG:
                    failures.append(f"{label}: joint margin only {margin:.1f}deg")
                    continue
                blocked = self.blend_contacts(attention, candidate, obstacles) | self.blend_contacts(candidate, hover_joints, obstacles)
                if blocked:
                    failures.append(f"{label}: blend hits {', '.join(sorted(blocked))}")
                    continue
                return candidate, center
        raise RuntimeError("no clear raise way point:\n    " + "\n    ".join(failures))

    def _plan_pick(self, object_position: np.ndarray) -> Plan:
        centers = self.centers(object_position)
        joints: dict[str, np.ndarray] = {}
        # The grasp is the most constrained pose: solve it first from the reference
        # posture, then walk backwards off it to the standoff and up to the hover.
        joints["grasp"] = solve_pose_ik(self.model, self.side, centers["grasp"], self.orientation, C.ARM_SEED[self.side])
        joints["pregrasp"] = self._walk(centers["grasp"], centers["pregrasp"], joints["grasp"], 6)[-1]
        joints["hover"] = self._walk(centers["pregrasp"], centers["hover"], joints["pregrasp"], 6)[-1]
        # Way point between the attention stance and the hover: the same hand pose
        # lifted straight up above where the hand hangs at attention, so the arm rises
        # first and only then travels over the table (a direct blend from the hanging
        # pose sweeps the fingers through whatever stands between, e.g. the basket).
        joints["raise"], centers["raise"] = self.find_raise(joints["hover"], centers["hover"])
        joints["ready"] = joints["hover"]
        joints["lift"] = solve_pose_ik(self.model, self.side, centers["lift"], self.orientation, joints["grasp"])
        for phase in ("pregrasp", "grasp"):
            hits = self.hand_contacts(joints[phase], self.scene.table_geoms)
            if hits:
                raise RuntimeError(f"{phase} hand would hit the floor ({', '.join(sorted(hits))})")
        clearance = self.fingertip_floor_clearance(joints["grasp"])
        if clearance < C.MIN_FLOOR_CLEARANCE:
            raise RuntimeError(f"grasp fingertip floor clearance only {clearance*1000:.1f}mm")

        return Plan(joints, {}, centers, side=self.side)

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
        # Only the place side is re-derived. At carry time `object_position` is the
        # lifted object, so the grasp-side centres from `centers()` no longer describe
        # the poses already executed; keep the originals and walk from the real lift.
        lift_center = plan.centers["lift"]
        feasible = []
        for yaw in C.PLACE_YAW_CANDIDATES_DEG:
            centers_yaw = {**plan.centers, **{
                k: v for k, v in self.centers(object_position, yaw, held_offset).items() if k in ("transfer", "lower")
            }}
            for strategy, route in self.transfer_route_candidates(lift_center, centers_yaw["transfer"]):
                try:
                    transfer_path, seed = [], joints["lift"]
                    for segment_index, (start, end) in enumerate(zip(route, route[1:])):
                        segment = self._walk(start, end, seed, C.CARRY_PATH_STEPS,
                            orientation_at=lambda f, yaw=yaw, first=segment_index == 0:
                                rotation_z(yaw * f if first else yaw) @ self.orientation)
                        transfer_path.extend(segment)
                        seed = segment[-1]
                    lower_path = self._walk(centers_yaw["transfer"], centers_yaw["lower"], seed, C.CARRY_PATH_STEPS,
                        orientation_at=lambda f, yaw=yaw: rotation_z(yaw) @ self.orientation)
                except RuntimeError as error:
                    failures.append(f"yaw {yaw:+.0f} {strategy}: {error}")
                    continue
                hits = self.hand_contacts(lower_path[-1], self.scene.basket_geoms)
                margin = min(self.joint_margin_degrees(q) for q in (*transfer_path, *lower_path))
                if hits or margin < C.MIN_JOINT_MARGIN_DEG:
                    failures.append(f"yaw {yaw:+.0f} {strategy}: collision or joint margin {margin:.1f}deg")
                    continue
                length = sum(float(np.linalg.norm(b - a)) for a, b in zip(route, route[1:]))
                feasible.append((length + 0.01 / margin, -margin, yaw, strategy, centers_yaw, transfer_path, lower_path))
        if feasible:
            _, _, yaw, strategy, centers_yaw, transfer_path, lower_path = min(feasible, key=lambda item: item[:2])
            joints["transfer"], joints["lower"] = transfer_path[-1], lower_path[-1]
            plan.paths = {"transfer": transfer_path, "lower": lower_path}
            plan.centers, plan.place_yaw_deg, plan.route_strategy = centers_yaw, yaw, strategy
            self.validate(plan)
            return plan
        raise RuntimeError("no reachable transfer/set-down pose at any hand yaw:\n  " + "\n  ".join(failures))

    @staticmethod
    def transfer_route_candidates(start: np.ndarray, target: np.ndarray) -> list[tuple[str, list[np.ndarray]]]:
        """Try a direct route and geometry-derived detours around the pedestal."""
        start, target = np.asarray(start, float), np.asarray(target, float)
        delta = target[:2] - start[:2]
        distance = float(np.linalg.norm(delta))
        direct = [("direct", [start.copy(), target.copy()])]
        if distance < 1e-9:
            return direct
        t = float(np.clip(-np.dot(start[:2], delta) / np.dot(delta, delta), 0.0, 1.0))
        nearest = start[:2] + t * delta
        if np.linalg.norm(nearest) >= C.TRANSFER_BASE_CLEARANCE and distance <= C.TRANSFER_LONG_PATH_M:
            return direct
        direction = delta / distance
        perpendicular = np.array([-direction[1], direction[0]])
        midpoint = 0.5 * (start[:2] + target[:2])
        offset = max(0.25 * distance, C.TRANSFER_BASE_CLEARANCE - float(np.linalg.norm(midpoint)) + 0.04)
        routes = []
        for label, xy in (("detour_left", midpoint + offset * perpendicular), ("detour_right", midpoint - offset * perpendicular)):
            waypoint = np.array([xy[0], xy[1], max(start[2], target[2])])
            routes.append((label, [start.copy(), waypoint, target.copy()]))
        return direct + routes

    def validate(self, plan: Plan) -> None:
        """Reject malformed, out-of-range or inaccurate endpoint solutions."""
        ids = np.array([self.model.joint(name).id for name in C.ARM_JOINTS[self.side]])
        lower, upper = self.model.jnt_range[ids].T
        for phase, joints in plan.joints.items():
            values = np.asarray(joints, dtype=float)
            if values.shape != (7,) or not np.all(np.isfinite(values)):
                raise RuntimeError(f"invalid {phase} joint target: expected 7 finite values")
            if np.any(values < lower) or np.any(values > upper):
                raise RuntimeError(f"{phase} joint target exceeds OpenArm v1 limits")
            reached, rotation = wrist_frame(self.model, self.side, values)
            desired_rotation = (
                rotation_z(plan.place_yaw_deg) @ self.orientation
                if phase in ("transfer", "lower")
                else self.orientation
            )
            position_error = float(np.linalg.norm(reached - plan.centers[phase]))
            rotation_error = float(np.linalg.norm(orientation_error(rotation, desired_rotation)))
            if position_error > C.IK_POSITION_TOLERANCE or rotation_error > C.IK_ROTATION_TOLERANCE:
                raise RuntimeError(
                    f"{phase} FK validation failed: position {position_error*1000:.1f}mm, "
                    f"orientation {rotation_error:.3f}rad"
                )

    def _pregrasp_data(self, arm_joints: np.ndarray, closed: bool = False) -> mujoco.MjData:
        """Throwaway MjData with the active arm at `arm_joints` and the hand pre-shaped
        (thumb opposed, fingers part-closed) or, with `closed`, a fist as at attention."""
        scene, model = self.scene, self.check_model
        data = mujoco.MjData(model)
        data.qpos[:] = scene.data.qpos
        side = self.side
        data.qpos[scene.arm_qpos[side]] = arm_joints
        if closed:
            data.qpos[scene.hand_qpos[side]] = scene.closed_hand[side]
        else:
            for name, actuator in scene.finger_actuator[side].items():
                joint = model.actuator_trnid[actuator, 0]
                opened, closed_value = scene.open_ctrl[side][name], scene.closed_ctrl[side][name]
                data.qpos[model.jnt_qposadr[joint]] = opened + C.GRASP_CLOSURE_FRACTION * (closed_value - opened)
            yaw_actuator, _, opposed = scene.thumb_yaw[side]
            data.qpos[model.jnt_qposadr[model.actuator_trnid[yaw_actuator, 0]]] = opposed
        # Poses and contacts are all the callers read; skip the dynamics.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        return data

    def hand_contacts(
        self, arm_joints: np.ndarray, target_geoms: set[int], closed: bool = False, clearance: float = 0.0
    ) -> set[str]:
        """Active-hand bodies (pre-shaped, or a fist with `closed`) intersecting any target
        geometry, or coming within `clearance` of it (up to PATH_CLEARANCE)."""
        scene, model = self.scene, self.model
        data = self._pregrasp_data(arm_joints, closed)
        hits: set[str] = set()
        for contact in data.contact[: data.ncon]:
            if contact.geom1 not in target_geoms and contact.geom2 not in target_geoms:
                continue
            other = contact.geom2 if contact.geom1 in target_geoms else contact.geom1
            if scene.hand_side(other) == self.side and float(contact.dist) < clearance:
                hits.add(model.body(int(model.geom_bodyid[other])).name)
        return hits

    def blend_contacts(self, start: np.ndarray, end: np.ndarray, target_geoms: set[int], samples: int = 40) -> set[str]:
        """Fist bodies that intersect `target_geoms` anywhere along a straight joint-space
        blend from `start` to `end` (what Executor.move_to executes; the hand stays a
        fist until the reach).
        The executed arm lags the commanded blend by a few mm, so the check demands
        PATH_CLEARANCE of room rather than mere non-penetration."""
        for fraction in np.linspace(0.0, 1.0, samples + 1)[1:]:
            hits = self.hand_contacts(start + (end - start) * fraction, target_geoms, closed=True, clearance=C.PATH_CLEARANCE)
            if hits:
                return hits  # the first blocked sample is reason enough to reject the blend
        return set()

    def fingertip_floor_clearance(self, arm_joints: np.ndarray) -> float:
        data = self._pregrasp_data(arm_joints)
        tips = [
            self.model.site(f"{HAND_PREFIX}{self.side}_{self.side}_{finger}_tip").id
            for finger in C.FINGER_NAMES
        ]
        return float(np.min(data.site_xpos[tips, 2]))

    # ------------------------------------------------------------------ debugging
    def describe(self, plan: Plan) -> str:
        """Table of phase -> target wrist position, FK of the solution, position error, wrist pitch."""
        lines = [f"{'phase':<10}{'target (x y z)':<28}{'reached (x y z)':<28}{'err mm':>8}{'wrist bend':>13}"]
        for phase in PHASE_ORDER:
            target = plan.centers[phase]
            reached, _ = wrist_frame(self.model, self.side, plan.joints[phase])
            error = np.linalg.norm(reached - target) * 1000
            pitch = np.degrees(np.hypot(*(plan.joints[phase][i] for i in C.WRIST_BEND_INDICES)))
            lines.append(
                f"{phase:<10}{np.array2string(target, precision=3):<28}{np.array2string(reached, precision=3):<28}"
                f"{error:>8.1f}{pitch:>12.1f}"
            )
        lines.append(f"grasp yaw: {plan.grasp_yaw_deg:+.0f} deg, place yaw: {plan.place_yaw_deg:+.0f} deg")
        return "\n".join(lines)
