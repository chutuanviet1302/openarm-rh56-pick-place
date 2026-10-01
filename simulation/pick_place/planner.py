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
    # Heading id for Demo's physics-retry exclusion list (yaw + 1000 * tilt index)
    # and the oblique tilt used, if any (GraspPlanner._grasp_tilts).
    grasp_key: float = 0.0
    grasp_tilt: tuple[str, float] | None = None
    place_yaw_deg: float = 0.0
    route_strategy: str = "direct"
    side: str = "right"
    # Twist-lift (see GraspPlanner._twist_walk): the hand turns this far about the
    # can's vertical axis between grasp and lift. `grasp_orientation` is the hand at
    # the grasp itself (the planner's `orientation` is the lifted, turned one).
    lift_twist_deg: float = 0.0
    grasp_orientation: np.ndarray | None = None
    # Grasp-library heading grasps: how far the jaw runs off square to the object's
    # axis (GraspPlanner._yaw_candidates).
    jaw_offset_deg: float = 0.0
    # Hand orientation per phase where it differs from the planner's `orientation`
    # (twist-lift: the grasp and the part-turned standoff).
    phase_orientations: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, phase: str) -> np.ndarray:
        return self.joints[phase]


class GraspPlanner:
    def __init__(self, scene: Scene, side: str = "right", *, use_seed_bank: bool = True) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        # `use_seed_bank=False`: only the two reference IK seeds (see _grasp_solutions).
        self.use_seed_bank = use_seed_bank
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
        # Oblique tilt of the heading being planned (None = reference grasp).
        self.grasp_tilt: tuple[str, float] | None = None
        self._local_jaw: tuple[np.ndarray, np.ndarray] | None = None
        self._local_jaw_fraction: float | None = None
        # Object width across the jaw for the heading being planned (off-square grasps
        # of a cylinder span more than its diameter); None = scene.object_extents().
        self.jaw_width: float | None = None
        # Largest jaw offset off square-on to try (None: all of GRASP_JAW_OFFSETS_DEG).
        # Demo tries square-on grasps at every drop spot before any offset one.
        self.max_jaw_offset_deg: float | None = None

    # ------------------------------------------------------------------ hand geometry
    def local_jaw_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean fingertip and thumb-tip offsets from the wrist, in the wrist's own frame,
        with the hand pre-shaped (thumb opposed, fingers at GRASP_CLOSURE_FRACTION).

        Measured once by forward kinematics on a throwaway MjData, so the targets follow
        from where this particular hand's jaws actually end up.
        """
        fraction = self.scene.grasp_closure_fraction
        if self._local_jaw is None or self._local_jaw_fraction != fraction:
            self._local_jaw = self.scene.jaw_offsets_at(self.side)
            self._local_jaw_fraction = fraction
        return self._local_jaw

    def jaw_offsets(self, orientation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(fingertip offset, thumb-tip offset) from the wrist in world axes for a wrist orientation."""
        fingers, thumb = self.local_jaw_offsets()
        return orientation @ fingers, orientation @ thumb

    # ------------------------------------------------------------------ targets
    def centers(
        self, object_position: np.ndarray, place_yaw_deg: float = 0.0, held_offset: np.ndarray | None = None,
        place_floor: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Wrist targets per phase, derived from the object, the basket and the jaw.

        `place_yaw_deg` turns the hand about world z on the release side only; the
        grasp itself always uses the straight-wrist orientation. `held_offset` is the
        object's *measured* position relative to the wrist (world axes, at the grasp
        orientation) once it is in hand; when given, the set-down is planned from where
        the object really sits instead of from the nominal jaw centre. `place_floor`
        overrides the set-down surface: the scene's one basket floor by default, or a
        bare table point when retrieving an object back out of that basket
        (simulation/pick_place/retrieve.py) -- the object still has to clear the same
        basket's rim on the way out, so only the floor used for the final descent
        changes, not the lift height.
        """
        scene = self.scene
        bottle = np.asarray(object_position, dtype=float)
        width, height = scene.object_extents()
        if self.jaw_width is not None:
            width = self.jaw_width
        offset, thumb_offset = self.jaw_offsets(self.orientation)

        # Straddle the object with the jaw centred on it in all three axes.
        jaw = 0.5 * (offset + thumb_offset)
        jaw_line = thumb_offset - offset
        jaw_axis = jaw_line / np.linalg.norm(jaw_line)
        # The height and along-jaw biases were tuned for the steep top grasp (lift the
        # grip so the fingertips clear the table; the left jaw shifted 30% toward the
        # fingers). Applied to an oblique grasp they put the left hand's fingertips at
        # the can's lid -- it closed above it and had to re-grasp (2026-09-24) -- so
        # oblique grasps use their own values.
        oblique = self.grasp_tilt is not None
        height_bias = C.OBLIQUE_GRASP_HEIGHT_BIAS if oblique else C.GRASP_HEIGHT_BIAS
        target = scene.grasp_target
        if target is not None and target.entry.height_bias is not None:
            height_bias = target.entry.height_bias
        finger_bias = C.OBLIQUE_JAW_BIAS_TOWARD_FINGERS if oblique else C.JAW_BIAS_TOWARD_FINGERS[self.side]
        grasp = (
            bottle
            - jaw
            + C.JAW_AXIS_BIAS * jaw_axis
            - finger_bias * jaw_line
            + np.array([0.0, 0.0, height_bias])
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
        # Oblique grasps on the work platform let the can settle ~1cm in the hand on
        # the way up (lift ended 4.1-4.2cm over the rim for a 5cm requirement,
        # 2026-09-24), so the lift aims correspondingly higher there.
        margin = C.CARRY_CLEARANCE_MARGIN + (C.OBLIQUE_LIFT_EXTRA if scene.work_surface_z > 0.0 else 0.0)
        lift_height = scene.carry_bottom_z() + margin + 0.5 * height - bottle[2]
        lift = grasp + np.array([0.0, 0.0, lift_height])

        drop = (
            (scene.basket_floor() if place_floor is None else np.asarray(place_floor, dtype=float))
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

    def plan(
        self, object_position: np.ndarray, exclude_yaws_deg: tuple[float, ...] = (),
        place_floor: np.ndarray | None = None,
    ) -> Plan:
        """Solve the whole waypoint chain. Raises RuntimeError with the phase that failed.

        Tries each grasp heading in C.GRASP_YAW_CANDIDATES_DEG (0 first, the reference
        posture) and keeps the first whose grasp, standoff, hover and lift all solve.
        `exclude_yaws_deg` skips headings already tried and found wanting in physics
        (the executor's contact / proof-lift checks), so a retry picks another.
        `place_floor` is forwarded to `centers()`: the scene's one basket by default,
        or a bare table point when retrieving an object back out of it. Oblique grasp
        headings (_grasp_tilts) are tried only after every reference heading failed.
        """
        failures = [f"grasp heading key {key:+.0f}: failed in physics, not retried" for key in exclude_yaws_deg]
        # Every tilt's square-on headings before any off-square one (stable sort: with
        # no grasp-library heading every offset is 0 and the order is unchanged).
        candidates = sorted(
            ((tilt_index, tilt, yaw, jaw_offset) for tilt_index, tilt in self._grasp_tilts()
             for yaw, jaw_offset in self._yaw_candidates(tilt)),
            key=lambda item: abs(item[3]),
        )
        target = self.scene.grasp_target
        for tilt_index, tilt, yaw, jaw_offset in candidates:
            # Key of this heading for exclude_yaws_deg: the yaw itself for the
            # reference (untilted) grasp, so existing callers are unchanged.
            key = yaw + 1000.0 * tilt_index
            if key in exclude_yaws_deg:
                continue
            label = f"grasp yaw {yaw:+.0f}" + (f" tilt {tilt[0]}{tilt[1]:+.0f}" if tilt else "")
            if jaw_offset:
                label += f" jaw {jaw_offset:+.0f} off square"
            self.orientation = rotation_z(yaw) @ self._tilted(tilt)
            self.grasp_tilt = tilt
            self.jaw_width = None
            if target is not None:
                self.jaw_width = target.entry.width / np.cos(np.radians(jaw_offset))
                try:
                    self.scene.fit_closure_fraction(self.jaw_width, self.side)
                except RuntimeError as error:
                    failures.append(f"{label}: {error}")
                    continue
            try:
                plan = self._plan_pick(object_position)
            except RuntimeError as error:
                failures.append(f"{label}: {error}")
                continue
            plan.grasp_yaw_deg, plan.grasp_key, plan.grasp_tilt = yaw, key, tilt
            plan.jaw_offset_deg = float(jaw_offset)
            try:
                self.plan_place(plan, object_position, place_floor=place_floor)
            except RuntimeError as error:
                failures.append(f"{label}: {error}")
                continue
            return plan
        self.orientation = self.base_orientation
        self.grasp_tilt = None
        raise RuntimeError("no reachable grasp at any hand yaw:\n  " + "\n  ".join(failures))

    def _grasp_tilts(self) -> list[tuple[int, tuple[str, float] | None]]:
        """(index, tilt) pairs to try: the reference grasp first, then the oblique ones.

        The reference grasp comes from the natural straight-wrist posture: fingers
        steeply down, fine on the table top. With the work raised (work platform,
        2026-09-24: 10cm) that grasp pins joints near the shoulder, while an oblique
        hand -- fingers ~45deg below horizontal, pointing forward and in -- reaches
        the centre basket with 25deg of joint margin on both arms. Tilts are about the
        world x/y axes applied to the reference orientation; x flips sign for the
        left arm (mirror image)."""
        mirror = 1.0 if self.side == "right" else -1.0
        tilts = [(axis, deg * (mirror if axis == "x" else 1.0)) for axis, deg in C.GRASP_TILT_CANDIDATES]
        if not self.use_seed_bank:
            return [(0, None)]
        oblique = []
        for i, tilt in enumerate(tilts):
            fingers, thumb = self.jaw_offsets(self._tilted(tilt))
            if abs(float(thumb[2] - fingers[2])) <= C.MAX_OBLIQUE_JAW_DZ:
                oblique.append((i + 1, tilt))
        # On the raised work platform the steep reference grasp can still plan, but at
        # a 41/36deg wrist bend it missed the can in physics and shoved it aside, so
        # the oblique grasps -- the natural ones up there -- go first.
        if self.scene.work_surface_z > 0.0:
            return oblique + [(0, None)]
        return [(0, None)] + oblique

    def _yaw_candidates(self, tilt: tuple[str, float] | None) -> list[tuple[float, float]]:
        """(hand yaw, jaw offset off square) pairs to try for one grasp tilt.

        Round seen from above (no grasp target, or the library gives no jaw heading):
        C.GRASP_YAW_CANDIDATES_DEG as always, offset 0. Otherwise the jaw must cross
        the library's axis (a lying can's, a pear's long axis): the yaw that turns
        this tilt's jaw line onto the square-on heading, that yaw + 180 (the object is
        just as graspable from its other side), each shifted by
        C.GRASP_JAW_OFFSETS_DEG; per offset, smallest turn from the reference first."""
        target = self.scene.grasp_target
        if target is None or target.jaw_heading_deg is None:
            return [(yaw, 0.0) for yaw in C.GRASP_YAW_CANDIDATES_DEG]
        fingers, thumb = self.jaw_offsets(self._tilted(tilt))
        line = thumb - fingers
        current = float(np.degrees(np.arctan2(line[1], line[0])))
        pairs = {}
        offsets = target.entry.jaw_offsets_deg or C.GRASP_JAW_OFFSETS_DEG
        for base in (target.jaw_heading_deg - current, target.jaw_heading_deg - current + 180.0):
            for offset in offsets:
                yaw = round((base + offset + 180.0) % 360.0 - 180.0, 1)
                pairs.setdefault(yaw, offset)
        if self.max_jaw_offset_deg is not None:
            pairs = {yaw: offset for yaw, offset in pairs.items() if abs(offset) <= self.max_jaw_offset_deg}
        return sorted(pairs.items(), key=lambda item: (abs(item[1]), abs(item[0])))

    def _tilted(self, tilt: tuple[str, float] | None) -> np.ndarray:
        if tilt is None:
            return self.base_orientation
        axis, deg = tilt
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        r = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == "x" else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return r @ self.base_orientation

    def plan_pick(self, object_position: np.ndarray) -> Plan:
        """Plan only through proof-lift; used by the bimanual route preflight."""
        failures = []
        self.grasp_tilt = None
        for yaw, _ in self._yaw_candidates(None):
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

        Because the first candidate over the floor wins, this pins the plan's *minimum*
        joint margin just above MIN_JOINT_MARGIN_DEG: mapped across the reachable table,
        every layout that planned at all reported the same 3.3 degrees. So that number
        describes where the accept test sits, not how hard the layout is -- judge a
        layout by the margin at its grasp pose instead, which is the constrained one.
        Taking the roomiest candidate here rather than the first was tried and is worse:
        it moves the raise away from the hover, the attention -> raise -> hover blend
        sweeps differently, and two trials lost the grasp outright (17/20 -> 16/20, and
        the mirrored left-arm case failed too).

        Returns (joints, centre); raises RuntimeError when nothing is clear."""
        hanging = self.scene.wrist_position_at(self.side, self.scene.attention_pose[self.side])
        attention = self.scene.attention_pose[self.side]
        obstacles = self.scene.basket_geoms | {self.scene.object_geom} | self.scene.table_geoms | self.scene.other_object_geoms()
        failures = []
        # Same fallback idea as Demo's centring retry (demo.py): a chain seed can sit
        # in a narrow IK basin even when the target is solvable from the reference
        # posture. Seeds from both are tried; whichever returns first still passes the
        # same margin and blend-clearance checks, so the accepted pose is unchanged in
        # kind -- only more layouts get one. Two guards keep this from widening the
        # acceptance gate:
        #   - the fallback seed is only tried when the chain seed's own IK FAILS, not
        #     when it returned a pinched solution (a 0.0deg-margin raise candidate must
        #     keep being rejected -- the executable-layout guard in
        #     tests/test_bimanual_routing.py matches a raise-phase "joint margin only
        #     0.0deg" failure, and a fresh seed must not rescue it);
        #   - the grasp itself keeps its single ARM_SEED solve, so a grasp at a joint
        #     limit is still rejected exactly as before.
        seeds = (hover_joints, C.ARM_SEED[self.side])
        # Real-height robot (2026-09-24): returning from a centre basket, neither seed
        # reached any raise candidate. The seed bank is a last resort per candidate,
        # tried only when every seed above FAILED its IK (same guard as above).
        bank = self._seed_bank(C.PLACE_SEED_BANK_SIZE) if self.use_seed_bank else []
        # The raise point is only passed through, so for an oblique grasp it may hold
        # the reference (fingers-down) hand instead: near the shoulder the tilted hand
        # has no IK there (work platform, 2026-09-24). The blends either side are
        # collision-checked as always.
        orientations = [self.orientation]
        if not np.allclose(self.orientation, self.base_orientation):
            orientations.append(self.base_orientation)
        for raise_orientation in orientations:
            # Heights above the hover first (unchanged order for layouts that plan),
            # then heights above the hanging hand itself: with the work raised the
            # hover sits near shoulder height, where no raise point solves, and the
            # lower ones swept the hand into the platform's front edge.
            heights = [hover_center[2] + extra for extra in np.linspace(C.RAISE_ABOVE_HOVER, 0.0, 5)]
            heights += [hanging[2] + lift for lift in C.RAISE_ABOVE_HANGING]
            for height in heights:
                extra = height - hover_center[2]
                for dx, raw_dy in C.RAISE_XY_OFFSETS:
                    dy = raw_dy if self.side == "right" else -raw_dy
                    center = np.array([hanging[0] + dx, hanging[1] + dy, height])
                    label = f"+{extra*100:.0f}cm ({dx:+.2f},{dy:+.2f})"
                    candidate = None
                    seed_errors = []
                    for seed_index, seed in enumerate(seeds):
                        try:
                            candidate = solve_pose_ik(self.model, self.side, center, raise_orientation, seed)
                        except RuntimeError as error:
                            seed_errors.append(str(error).splitlines()[0])
                            continue
                        margin = self.joint_margin_degrees(candidate)
                        if margin < C.MIN_JOINT_MARGIN_DEG:
                            # A pinched solution from this seed is a REAL rejection: the
                            # next seed may not rescue it (the executable-layout guard in
                            # tests/test_bimanual_routing.py matches exactly this
                            # raise-phase "joint margin only 0.0deg" rejection).
                            if seed_index + 1 < len(seeds):
                                seed_errors.append(f"joint margin only {margin:.1f}deg; trying reference-posture seed")
                                candidate = None
                                continue
                            candidate = None
                            break
                        break
                    if candidate is None and len(seed_errors) == len(seeds) and all("IK failed" in e for e in seed_errors):
                        for seed in bank:
                            try:
                                q = solve_pose_ik(self.model, self.side, center, raise_orientation, seed,
                                                  max_iterations=C.GRASP_SEED_BANK_ITERATIONS)
                            except RuntimeError:
                                continue
                            if self.joint_margin_degrees(q) >= C.MIN_JOINT_MARGIN_DEG:
                                candidate = q
                                break
                    if candidate is None:
                        failures.append(f"{label}: {seed_errors[-1]}")
                        continue
                    blocked = self.blend_contacts(attention, candidate, obstacles) | self.blend_contacts(candidate, hover_joints, obstacles)
                    if blocked:
                        failures.append(f"{label}: blend hits {', '.join(sorted(blocked))}")
                        continue
                    return candidate, center
        raise RuntimeError("no clear raise way point:\n    " + "\n    ".join(failures))

    def _seed_bank(self, size: int) -> list[np.ndarray]:
        """Fixed (GRASP_SEED_BANK_RNG) joint seeds spread over the arm's ranges."""
        ids = np.array([self.model.joint(name).id for name in C.ARM_JOINTS[self.side]])
        lower, upper = self.model.jnt_range[ids].T
        rng = np.random.default_rng(C.GRASP_SEED_BANK_RNG)
        return [lower + (upper - lower) * rng.uniform(0.1, 0.9, len(lower)) for _ in range(size)]

    def _grasp_solutions(self, target: np.ndarray, use_bank: bool) -> list[np.ndarray]:
        """IK solutions for the grasp pose, in the order they should be tried.

        The two reference seeds come first (reference posture, then the natural
        straight-wrist posture -- the pair scripts/sweep_reach_map.py maps the grasp
        zone with), so layouts that already planned keep their exact solution. Then a
        fixed bank of seeds spread over the joint ranges, best joint margin first:
        near the table centreline and over a raised basket the reference seeds miss
        branches that exist -- 80 spread seeds found the left arm's grasp at
        (0.30, 0.00) 10cm up with an 11deg margin where both reference seeds failed
        (measured 2026-09-23). Solutions under MIN_JOINT_MARGIN_DEG are dropped.
        """
        natural = C.NATURAL_GRASP_JOINTS * (1.0 if self.side == "right" else C.MIRROR_JOINT_SIGNS)
        found: list[np.ndarray] = []

        def solve(seed: np.ndarray, iterations: int) -> np.ndarray | None:
            try:
                q = solve_pose_ik(self.model, self.side, target, self.orientation, seed, max_iterations=iterations)
            except RuntimeError:
                return None
            margin = self.joint_margin_degrees(q)
            if margin < C.MIN_JOINT_MARGIN_DEG:
                # Remembered so a pose that exists but pins a joint is reported as
                # such, not as "unreachable".
                self._low_margin = max(self._low_margin, margin)
                return None
            if any(np.max(np.abs(q - other)) < C.GRASP_SEED_DUPLICATE_RAD for other in found):
                return None
            return q

        for seed in (C.ARM_SEED[self.side], natural):
            q = solve(seed, C.IK_MAX_ITERATIONS)
            if q is not None:
                found.append(q)
        if not use_bank:
            return found
        ids = np.array([self.model.joint(name).id for name in C.ARM_JOINTS[self.side]])
        lower, upper = self.model.jnt_range[ids].T
        rng = np.random.default_rng(C.GRASP_SEED_BANK_RNG)
        bank = []
        for _ in range(C.GRASP_SEED_BANK_SIZE):
            q = solve(lower + (upper - lower) * rng.uniform(0.1, 0.9, len(lower)), C.GRASP_SEED_BANK_ITERATIONS)
            if q is not None:
                bank.append(q)
                found.append(q)
        bank.sort(key=self.joint_margin_degrees, reverse=True)
        return bank

    def _plan_pick(self, object_position: np.ndarray) -> Plan:
        base_centers = self.centers(object_position)
        base_orientation = self.orientation
        # The seed bank costs ~40 IK solves per heading, so it only runs when the
        # reference seeds found nothing that chains (a plain pick never pays for it).
        errors, tried = [], 0
        self._low_margin = -np.inf
        for use_bank in (False, True) if self.use_seed_bank else (False,):
            # A failed twist-lift attempt leaves its turned heading on self.orientation;
            # the grasp itself is always solved at the plan's grasp heading.
            self.orientation = base_orientation
            solutions = self._grasp_solutions(base_centers["grasp"], use_bank)
            tried += len(solutions)
            for grasp_joints in solutions[: C.GRASP_CHAIN_ATTEMPTS]:
                self.orientation = base_orientation
                try:
                    return self._chain_from_grasp(object_position, {k: v.copy() for k, v in base_centers.items()}, grasp_joints)
                except RuntimeError as error:
                    errors.append(str(error).splitlines()[0])
        self.orientation = base_orientation
        if not tried and np.isfinite(self._low_margin):
            raise RuntimeError(f"grasp joint margin only {self._low_margin:.1f}deg")
        if not tried:
            raise RuntimeError(f"IK failed for {self.side} target {base_centers['grasp'].tolist()}")
        raise RuntimeError(f"{tried} grasp solution(s), none chains: {errors[0]}")

    def _chain_from_grasp(self, object_position: np.ndarray, centers: dict, grasp_joints: np.ndarray) -> Plan:
        """Standoff, hover, raise and lift walked off one grasp solution (same branch)."""
        joints: dict[str, np.ndarray] = {"grasp": grasp_joints}
        grasp_orientation, twist = self.orientation, 0.0
        self._twist_phase_orientations = {}
        try:
            joints["pregrasp"] = self._walk(centers["grasp"], centers["pregrasp"], joints["grasp"], 6)[-1]
            joints["hover"] = self._walk(centers["pregrasp"], centers["hover"], joints["pregrasp"], 6)[-1]
        except RuntimeError as straight_error:
            twisted = self._twist_approach(object_position, centers, joints)
            if twisted is None:
                raise straight_error
            twist = twisted
        # Way point between the attention stance and the hover: the same hand pose
        # lifted straight up above where the hand hangs at attention, so the arm rises
        # first and only then travels over the table (a direct blend from the hanging
        # pose sweeps the fingers through whatever stands between, e.g. the basket).
        joints["raise"], centers["raise"] = self.find_raise(joints["hover"], centers["hover"])
        joints["ready"] = joints["hover"]
        if not twist:
            joints["lift"] = solve_pose_ik(self.model, self.side, centers["lift"], self.orientation, joints["grasp"])
        for phase in ("pregrasp", "hover", "lift"):
            margin = self.joint_margin_degrees(joints[phase])
            if margin < C.MIN_JOINT_MARGIN_DEG:
                raise RuntimeError(f"{phase} joint margin only {margin:.1f}deg")
        # Also checked against the basket (and its stand): a pick that starts *inside*
        # the basket (retrieval, simulation/pick_place/retrieve.py) must clear its
        # walls on the way in. For every other pick the basket sits far away.
        obstacles = self.scene.table_geoms | self.scene.basket_geoms | self.scene.other_object_geoms()
        for phase in ("pregrasp", "grasp"):
            hits = self.hand_contacts(joints[phase], obstacles)
            if hits:
                raise RuntimeError(f"{phase} hand would hit an obstacle ({', '.join(sorted(hits))})")
        # Grasp-library objects: the palm must not sit in the object at the grasp. On a
        # tall object (191 mm mustard bottle) the palm of a top grasp came down on the
        # cap and tipped the bottle 16 deg before the fingers closed (2026-09-28).
        if self.scene.grasp_target is not None:
            for phase in ("pregrasp", "grasp"):
                depth = self.palm_in_object(joints[phase])
                if depth is not None:
                    raise RuntimeError(f"{phase} palm would press {depth*1000:.0f}mm into the object")
        clearance = self.fingertip_floor_clearance(joints["grasp"])
        if clearance < C.MIN_FLOOR_CLEARANCE:
            raise RuntimeError(f"grasp fingertip floor clearance only {clearance*1000:.1f}mm")
        # The arm itself must keep off the torso/pedestal too (the executor aborts on
        # any touch): an oblique left grasp bent the wrist 61deg and link5 grazed the
        # torso on the proof lift (2026-09-24). Checked at the phase poses and along
        # the grasp -> lift blend the proof lift and lift follow.
        blend = [joints["grasp"] + (joints["lift"] - joints["grasp"]) * f for f in np.linspace(0.0, 1.0, 6)]
        for label, q in [(k, joints[k]) for k in ("pregrasp", "hover", "lift")] + [("grasp->lift", q) for q in blend]:
            gap = self.arm_body_clearance(q)
            if gap < C.ARM_BODY_CLEARANCE:
                raise RuntimeError(f"{label}: arm {gap*1000:.1f}mm from the robot's torso/pedestal")

        return Plan(joints, {}, centers, side=self.side, lift_twist_deg=twist, grasp_orientation=grasp_orientation,
            phase_orientations=dict(self._twist_phase_orientations),
        )

    def _twist_approach(self, object_position: np.ndarray, centers: dict, joints: dict) -> float | None:
        """Fallback when the straight approach/lift runs an arm into its joint limits.

        On the real OpenArm v1 mount a can near the table centreline sits at the edge
        of each arm's reach: the left arm can close on it, but lifting at a fixed hand
        heading drives joint5 into its 90deg stop (or joint6 into 45deg) within a few
        cm (measured 2026-09-23, can at (0.197, 0.045)). Turning the hand about the
        can's own vertical axis while rising keeps the can upright and walks the
        forearm roll back off the stop. The approach is the same path reversed: the
        open hand comes straight down over the can, untwisting to the grasp heading.
        Sets joints/centers pregrasp, hover and lift, and `self.orientation` to the
        lifted heading. Returns the twist in degrees, or None.
        """
        grasp_center = centers["grasp"]
        pivot = np.array([object_position[0], object_position[1], grasp_center[2]])
        rise = max(float(centers["lift"][2] - grasp_center[2]) + C.TWIST_LIFT_EXTRA_RISE, C.APPROACH_STANDOFF)
        steps = C.TWIST_LIFT_STEPS
        obstacles = self.scene.table_geoms | self.scene.basket_geoms | self.scene.other_object_geoms()
        base = self.orientation
        for twist in C.TWIST_LIFT_CANDIDATES_DEG:
            path, targets, seed = [], [], joints["grasp"]
            try:
                for k in range(1, steps + 1):
                    f = k / steps
                    turn = rotation_z(twist * f)
                    target = pivot + turn @ (grasp_center - pivot) + np.array([0.0, 0.0, rise * f])
                    seed = solve_pose_ik(self.model, self.side, target, turn @ base, seed)
                    path.append(seed)
                    targets.append(target)
            except RuntimeError:
                continue
            if min(self.joint_margin_degrees(q) for q in path) < C.MIN_JOINT_MARGIN_DEG:
                continue
            if any(self.hand_contacts(q, obstacles) for q in path):
                continue
            # Standoff: back off along the fingers as a plain pick does when that still
            # solves -- straight down over the can, the open index finger lands on its
            # rim and shoves it (14mm measured). Only otherwise use a point on the
            # twist path half a standoff above the grasp.
            self._twist_phase_orientations = {"grasp": base}
            try:
                joints["pregrasp"] = self._walk(grasp_center, centers["pregrasp"], joints["grasp"], 12)[-1]
                if self.joint_margin_degrees(joints["pregrasp"]) < C.MIN_JOINT_MARGIN_DEG:
                    raise RuntimeError("standoff joint margin")
                self._twist_phase_orientations["pregrasp"] = base
            except RuntimeError:
                index = next(i for i, t in enumerate(targets) if t[2] - grasp_center[2] >= C.APPROACH_STANDOFF * 0.5)
                joints["pregrasp"], centers["pregrasp"] = path[index], targets[index]
                self._twist_phase_orientations["pregrasp"] = rotation_z(twist * (index + 1) / steps) @ base
            joints["lift"], centers["lift"] = path[-1], targets[-1]
            joints["hover"], centers["hover"] = path[-1], targets[-1]
            centers["ready"] = targets[-1]
            self.orientation = rotation_z(twist) @ base
            return float(twist)
        self.orientation = base
        return None

    def plan_place(
        self, plan: Plan, object_position: np.ndarray, held_offset: np.ndarray | None = None,
        place_floor: np.ndarray | None = None,
    ) -> Plan:
        """Solve transfer + set-down (in place, on `plan`) from the lift pose.

        The place side may turn the hand about the vertical: the can stays upright
        either way, and it lets the basket sit where the grasp orientation alone
        cannot reach. Every intermediate solution is kept as a carry waypoint.
        Called once at planning time with the nominal jaw, and again after the grasp
        with the measured `held_offset` so the object -- not the wrist -- lands on the
        basket centre. `place_floor` overrides the set-down surface (see `centers()`).
        """
        joints = plan.joints
        failures = []
        # Only the place side is re-derived. At carry time `object_position` is the
        # lifted object, so the grasp-side centres from `centers()` no longer describe
        # the poses already executed; keep the originals and walk from the real lift.
        lift_center = plan.centers["lift"]
        # The transfer is seeded from the lift pose to stay on the approach's IK
        # branch. That branch can be unable to fold the elbow across an inner-zone
        # carry where a second branch solves comfortably: on (0.14, -0.27) ->
        # (0.31, -0.06) every chained transfer waypoint raised "IK failed" while the
        # reference straight-wrist posture seeded margins of 3-19 deg on the very
        # same targets. First seed whose walk completes wins, so layouts that
        # already planned keep their exact carry path.
        transfer_seeds = (
            joints["lift"],
            C.NATURAL_GRASP_JOINTS if self.side == "right" else C.NATURAL_GRASP_JOINTS * C.MIRROR_JOINT_SIGNS,
        )
        feasible = []
        # Second pass only when the two reference seeds carry nowhere: a fixed bank of
        # seeds over the joint ranges (as for the grasp), since the lift pose's branch
        # can be unable to reach a set-down the arm reaches on another branch.
        # The bank pass walks each route backwards from the set-down end (solved from
        # the bank seed) to the lift: the lift's branch may not reach the set-down,
        # while the set-down's own branch reaches back to the lift height; the joint
        # blend below then joins the lift pose onto it (collision-checked).
        passes = [(transfer_seeds, False)]
        if self.use_seed_bank:
            passes.append((tuple(self._seed_bank(C.PLACE_SEED_BANK_SIZE)), True))
        for transfer_seeds, reverse in passes:
            if feasible:
                break
            for yaw in C.PLACE_YAW_CANDIDATES_DEG:
                centers_yaw = {**plan.centers, **{
                    k: v for k, v in self.centers(object_position, yaw, held_offset, place_floor).items()
                    if k in ("transfer", "lower")
                }}
                for strategy, route in self.transfer_route_candidates(lift_center, centers_yaw["transfer"]):
                    solved, seed_errors = None, []
                    for transfer_seed in transfer_seeds:
                        try:
                            if reverse:
                                transfer_path = self._reverse_route_walk(route, yaw, transfer_seed)
                                seed = transfer_path[-1]
                            else:
                                transfer_path, seed = [], transfer_seed
                                for segment_index, (start, end) in enumerate(zip(route, route[1:])):
                                    segment = self._walk(start, end, seed, C.CARRY_PATH_STEPS,
                                        orientation_at=lambda f, yaw=yaw, first=segment_index == 0:
                                            rotation_z(yaw * f if first else yaw) @ self.orientation)
                                    transfer_path.extend(segment)
                                    seed = segment[-1]
                            lower_path = self._walk(centers_yaw["transfer"], centers_yaw["lower"], seed, C.CARRY_PATH_STEPS,
                                orientation_at=lambda f, yaw=yaw: rotation_z(yaw) @ self.orientation)
                        except RuntimeError as error:
                            seed_errors.append(str(error).splitlines()[0])
                            continue
                        # A walk can complete on a pinched branch the margin test then
                        # rejects; the next seed may carry the same route through with
                        # room to spare, so acceptance is checked per seed here. The
                        # accepted path must also START from where the arm really is:
                        # a whole-path seed whose first waypoint jumps in joint space
                        # (measured 1.25 rad lift->wp0 on the centreline layout) makes
                        # the follower teleport the arm and shake the can loose -- the
                        # carry then fails "transfer too low" in physics even though
                        # every waypoint validated. The same-jump problem is fixed, not
                        # just rejected: the blend from the lift pose to that first
                        # waypoint is pre-solved in joint space (it is short in Cartesian
                        # terms -- the branch switch, not the wrist move, causes the
                        # jump) and prepended as a convergence segment.
                        jump = float(np.max(np.abs(transfer_path[0] - joints["lift"])))
                        if jump > C.MAX_SEED_JUMP_RAD:
                            blend = self.blend_transfer_segments(joints["lift"], transfer_path[0])
                            if blend is None:
                                seed_errors.append(f"first transfer waypoint jumps {jump:.2f}rad from the lift pose and no clear joint blend exists")
                                continue
                            transfer_path = blend + transfer_path
                        hits = self.hand_contacts(lower_path[-1], self.scene.basket_geoms)
                        margin = min(self.joint_margin_degrees(q) for q in (*transfer_path, *lower_path))
                        if hits or margin < C.MIN_JOINT_MARGIN_DEG:
                            seed_errors.append(f"collision or joint margin {margin:.1f}deg")
                            continue
                        # The whole carry keeps the arm off the torso too (left link5
                        # touched it mid-carry after a 61deg wrist-bend grasp,
                        # 2026-09-24); every other waypoint plus the ends.
                        carry = [*transfer_path, *lower_path]
                        gap = min(self.arm_body_clearance(q) for q in carry[::2] + [carry[-1]])
                        if gap < C.ARM_BODY_CLEARANCE:
                            seed_errors.append(f"carry passes {gap*1000:.1f}mm from the robot's torso")
                            continue
                        solved = (transfer_path, lower_path)
                        break
                    if solved is None:
                        failures.append(f"yaw {yaw:+.0f} {strategy}: {seed_errors[-1]}")
                        continue
                    transfer_path, lower_path = solved
                    margin = min(self.joint_margin_degrees(q) for q in (*transfer_path, *lower_path))
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

    def _reverse_route_walk(self, route: list[np.ndarray], yaw: float, seed: np.ndarray) -> list[np.ndarray]:
        """Forward-ordered carry waypoints for `route`, solved end-first from `seed`.
        Same waypoints and hand headings as the forward walk in plan_place (the first
        segment turns the hand by `yaw`, the rest hold it)."""
        end_orientation = rotation_z(yaw) @ self.orientation
        seed = solve_pose_ik(self.model, self.side, route[-1], end_orientation, seed)
        backwards = [seed]
        segments = list(enumerate(zip(route, route[1:])))
        for segment_index, (start, end) in reversed(segments):
            first = segment_index == 0
            # Walking end -> start: fraction f from the end is (1 - f) along the segment.
            points = self._walk(end, start, backwards[-1], C.CARRY_PATH_STEPS,
                orientation_at=lambda f, first=first: rotation_z(yaw * (1.0 - f) if first else yaw) @ self.orientation)
            backwards.extend(points)
        # backwards runs route[-1] ... route[0] (the lift itself last); drop the lift
        # point so the path starts one step out, as the forward walk does.
        return list(reversed(backwards[:-1]))

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
            if phase in ("transfer", "lower"):
                desired_rotation = rotation_z(plan.place_yaw_deg) @ self.orientation
            else:
                desired_rotation = plan.phase_orientations.get(phase, self.orientation)
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
        # One scratch MjData per planner, overwritten each call (every caller reads it
        # at once): allocating one per check was ~27% of a hand_contacts call.
        data = self.__dict__.get("_scratch")
        if data is None:
            data = self._scratch = mujoco.MjData(model)
        data.qpos[:] = scene.data.qpos
        side = self.side
        data.qpos[scene.arm_qpos[side]] = arm_joints
        if closed:
            data.qpos[scene.hand_qpos[side]] = scene.rest_hand[side]
        else:
            for name, actuator in scene.finger_actuator[side].items():
                joint = model.actuator_trnid[actuator, 0]
                opened, closed_value = scene.open_ctrl[side][name], scene.closed_ctrl[side][name]
                data.qpos[model.jnt_qposadr[joint]] = opened + scene.grasp_closure_fraction * (closed_value - opened)
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

    def palm_in_object(self, arm_joints: np.ndarray) -> float | None:
        """Deepest penetration (m) of the active hand's non-finger bodies (palm, base)
        into the pick object with the hand pre-shaped at `arm_joints`, or None."""
        scene = self.scene
        data = self._pregrasp_data(arm_joints)
        deepest = None
        for contact in data.contact[: data.ncon]:
            if scene.object_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == scene.object_geom else contact.geom1
            if scene.hand_side(other) != self.side or scene.finger_of(other, self.side) is not None:
                continue
            if float(contact.dist) < 0.0:
                deepest = max(deepest or 0.0, -float(contact.dist))
        return deepest

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

    def blend_transfer_segments(self, start_joints: np.ndarray, first_waypoint: np.ndarray, samples: int = 12) -> list[np.ndarray] | None:
        """Joint-space blend from the lift pose to an alternate-branch transfer start,
        as a list of intermediate joint targets the executor can `follow` segment by
        segment. The blend is only offered when it is short in Cartesian terms (the
        wrist barely moves across an IK branch switch) and stays collision-free with
        PATH_CLEARANCE of room; otherwise None, and the seed is rejected as before."""
        start_pos, _ = wrist_frame(self.model, self.side, start_joints)
        end_pos, _ = wrist_frame(self.model, self.side, first_waypoint)
        if float(np.linalg.norm(end_pos - start_pos)) > C.BRANCH_BLEND_MAX_CARTESIAN_M:
            return None
        obstacles = self.scene.table_geoms | self.scene.basket_geoms | self.scene.other_object_geoms()
        for fraction in np.linspace(0.0, 1.0, samples + 1)[1:]:
            joints = start_joints + (first_waypoint - start_joints) * fraction
            margin = self.joint_margin_degrees(joints)
            if margin < C.MIN_JOINT_MARGIN_DEG:
                return None
            if self.blend_contacts(start_joints + (first_waypoint - start_joints) * (fraction - 1.0 / samples), joints, obstacles):
                return None
        return [
            start_joints + (first_waypoint - start_joints) * f
            for f in np.linspace(0.0, 1.0, samples + 1)[1:-1]
        ]

    def arm_body_clearance(self, arm_joints: np.ndarray) -> float:
        """Smallest distance (m, capped at ARM_BODY_CLEARANCE_MAX) from this arm's links
        2..7 and hand to the robot's own torso/pedestal, with the arm at `arm_joints`."""
        scene, model = self.scene, self.model
        data = mujoco.MjData(model)
        data.qpos[:] = scene.data.qpos
        data.qpos[scene.arm_qpos[self.side]] = arm_joints
        mujoco.mj_kinematics(model, data)
        mounts = {f"openarm_{self.side}_link{i}" for i in (0, 1)}
        best = C.ARM_BODY_CLEARANCE_MAX
        fromto = np.zeros(6)
        for geom in range(model.ngeom):
            if scene.robot_side(geom) != self.side or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if (model.body(int(model.geom_bodyid[geom])).name or "") in mounts:
                continue
            for body_geom in scene.robot_body_geoms:
                best = min(best, mujoco.mj_geomDistance(model, data, geom, body_geom, best, fromto))
        return float(best)

    def fingertip_floor_clearance(self, arm_joints: np.ndarray) -> float:
        data = self._pregrasp_data(arm_joints)
        tips = [
            self.model.site(f"{HAND_PREFIX}{self.side}_{self.side}_{finger}_tip").id
            for finger in C.FINGER_NAMES
        ]
        # Measured from the surface the work happens on (table top or work platform).
        return float(np.min(data.site_xpos[tips, 2])) - self.scene.work_surface_z

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
