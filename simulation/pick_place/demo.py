"""The pick-and-place episode as an explicit sequence of phases.

    Demo(...)            builds Scene + GraspPlanner + Executor + EpisodeLog
    demo.run(viewer)     perceive -> plan -> ready -> reach -> grasp -> carry -> release
    demo.run(stop_after="grasp")   stop with the scene left in that phase's end state

Each `phase_*` method does one thing, checks one thing, and records its evidence in
`demo.log.values`; `run_trial()` turns that into a TrialResult.
"""

from __future__ import annotations

import numpy as np

from simulation.five_finger_model import BASKET_HALF_WIDTH, BASKET_POSITION_B, BASKET_WALL_THICKNESS, PICK_POSITION_A
from simulation.pick_place import config as C
from simulation.pick_place.episode import EpisodeLog, TrialResult
from simulation.pick_place.executor import Executor
from simulation.pick_place.kinematics import upright_tilt_degrees
from simulation.pick_place.planner import GraspPlanner, Plan
from simulation.pick_place.scene import Scene
from simulation.vision_detector import VisionDetector

PHASES = ("perceive", "plan", "ready", "reach", "grasp", "carry", "release")


class Demo:
    # A secure grasp needs the thumb opposing the fingers (an antipodal pinch), not just
    # fingers piled on one side. Tests assert this stays on.
    REQUIRE_THUMB_OPPOSITION = True

    def __init__(
        self,
        pick_position=PICK_POSITION_A,
        basket_position=BASKET_POSITION_B,
        *,
        perception: bool = False,
        perception_camera: str = "d435_head",
        verbose: bool = False,
        side: str = "right",
        scene: Scene | None = None,
        place_offset: tuple[float, float] | None = None,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.route_reason = "direct arm configured"
        # `scene` lets a caller continue an episode already in progress on the same
        # MjModel/MjData (simulation/pick_place/retrieve.py: grasping an object back
        # out of the basket a previous episode really left it in, walls and all,
        # instead of rebuilding a fresh table with no basket at that spot).
        self.scene = scene if scene is not None else Scene(pick_position, basket_position)
        self.planner = GraspPlanner(self.scene, side)
        self.executor = Executor(self.scene)
        self.executor.active_side = side
        # Release point relative to the basket centre (xy, metres). The basket stays
        # where the task puts it; this only moves where inside it the can is set down
        # (retrieve.py: toward the side the other arm can grasp from).
        self.place_offset = None if place_offset is None else np.array([*place_offset, 0.0], dtype=float)
        self.log = EpisodeLog(lambda: self.scene.data.time, verbose=verbose)
        # With perception on, the object's position comes from the camera (RGB-D ->
        # deprojection), never from the simulator state; the error against ground truth
        # is logged so a calibration or segmentation fault is visible in the report.
        self.perception = perception
        self.perception_camera = perception_camera
        self.perceived_position: np.ndarray | None = None
        self.perception_error_m: float | None = None
        self.plan: Plan | None = None
        self.light_grip_ctrl: np.ndarray | None = None
        self.failed_grasp_yaws: list[float] = []

    # ------------------------------------------------------------------ convenience views
    @property
    def model(self):
        return self.scene.model

    @property
    def data(self):
        return self.scene.data

    @property
    def episode(self) -> dict:
        return self.log.values

    @property
    def place_yaw_deg(self) -> float:
        return self.plan.place_yaw_deg if self.plan else 0.0

    def __getattr__(self, name):
        # Index tables and geometry live on the scene (model, data, hand_qpos, open_hand,
        # ee_site_id, bottle_qpos, attention_pose, grasp_orientation, ...).
        scene = self.__dict__.get("scene")
        if scene is not None and hasattr(scene, name):
            return getattr(scene, name)
        raise AttributeError(name)

    # ------------------------------------------------------------------ perception
    def object_position(self) -> np.ndarray:
        """Where the planner believes the object is."""
        if self.perception:
            if self.perceived_position is None:
                self.perceive_object()
            return self.perceived_position.copy()
        return self.scene.object_position()

    def perceive_object(self) -> np.ndarray:
        """RGB-D detection from the head camera; raises if the object is not seen.
        No silent fallback to ground truth: a run that cannot see its object must fail
        as a perception failure, not succeed by cheating."""
        result = VisionDetector(self.model, self.perception_camera).detect_object(self.data, render_annotation=False)
        if not result.found:
            raise RuntimeError(
                f"perception failed: object not found in camera '{self.perception_camera}' "
                f"({result.pixel_count} candidate pixels)"
            )
        truth = self.scene.object_position()
        self.perceived_position = np.asarray(result.pos_world, dtype=float)
        self.perception_error_m = float(np.linalg.norm(self.perceived_position[:2] - truth[:2]))
        self.log.record("perceived_position", self.perceived_position)
        self.log.record("perception_error_m", self.perception_error_m)
        return self.perceived_position

    # ------------------------------------------------------------------ grasp checks
    def fingers_not_pressing(self) -> list[str]:
        forces = self.scene.finger_contact_forces(self.side)
        missing = []
        # The thumb alone opposes the four fingers, so it carries the load: a thumb that
        # never built up real pressure (3.7N seen at a heading that then dropped the
        # can) is a grasp to let go of *before* the proof lift disturbs the object.
        if self.REQUIRE_THUMB_OPPOSITION and forces["thumb"] < C.GRASP_SECURE_MIN_THUMB_FORCE_N:
            missing.append("thumb")
        fingers = [name for name in C.FINGER_NAMES if name != "thumb"]
        pressing = [name for name in fingers if forces[name] >= C.GRASP_SECURE_MIN_FORCE_N]
        if len(pressing) < 2:
            missing.extend(name for name in fingers if name not in pressing)
        # On this hand the thumb closes against the index/middle side; a grip carried
        # by ring and pinky alone (index 0N, middle 0.7N seen) pinches the can off
        # centre and let it slip out on the lift.
        if all(forces[name] < C.GRASP_SECURE_MIN_OPPOSING_FORCE_N for name in ("index", "middle")):
            missing.extend(name for name in ("index", "middle") if name not in missing)
        return missing

    def grasp_is_secure(self) -> bool:
        """Force-based antipodal grasp: thumb opposed by at least two fingers."""
        return not self.fingers_not_pressing()

    # ------------------------------------------------------------------ phases
    def phase_perceive(self) -> None:
        if not self.perception:
            self.log.note("perception off: planning from the simulator's object pose")
            return
        self.perceive_object()
        self.log.note(
            f"object seen at {np.round(self.perceived_position, 3).tolist()} "
            f"(error vs ground truth {self.perception_error_m*1000:.1f}mm)"
        )

    def resolve_lift_from_here(self) -> None:
        """Re-solve the lift pose from the arm's current joints (after the proof lift).

        The 7-DOF arm reaches the planned lift wrist pose with a whole family of elbow
        positions; the plan's lift came off the *planned* grasp, the arm now sits at
        the *executed* one. The two landed 0.40rad apart (joint4 23deg, 2026-09-24,
        perception layout) and the joint-space move between them swung the hand and
        dropped the can. Seeding from here keeps the elbow where it is. The carry is
        re-planned from this lift pose right after (plan_place seeds from it). Not for
        a twist-lift plan, whose lift pose is the end of its own turning path."""
        from simulation.pick_place.kinematics import solve_pose_ik

        plan, scene, side = self.plan, self.scene, self.side
        if plan.lift_twist_deg:
            return
        here = self.data.ctrl[scene.arm_actuators[side]].copy()
        try:
            lift = solve_pose_ik(self.model, side, plan.centers["lift"], self.planner.orientation, here)
        except RuntimeError:
            return  # keep the planned lift pose
        if self.planner.joint_margin_degrees(lift) >= C.MIN_JOINT_MARGIN_DEG:
            plan.joints["lift"] = lift

    def lift_straight_up(self) -> None:
        """From the proof-lift pose up to the planned lift height along a straight
        vertical line, hand orientation held. A joint-space move there swings the hand
        on a curve; an oblique grasp on the work platform dropped the can doing that
        (2026-09-24). Falls back to the planned joint-space move if the line has no IK."""
        ex, plan, scene, side = self.executor, self.plan, self.scene, self.side
        arm = f"{side}_arm"
        if plan.lift_twist_deg:
            ex.move_to({arm: plan["lift"]}, C.MOVE_TO_LIFT)
            return
        if scene.work_surface_z <= 0.0:
            # On the table top the straight line landed the lift 0.40rad from the
            # planned one and the re-planned carry dropped the can (default
            # perception layout); the elbow-continuous re-solve works there.
            self.resolve_lift_from_here()
            ex.move_to({arm: plan["lift"]}, C.MOVE_TO_LIFT)
            return
        here = self.data.ctrl[scene.arm_actuators[side]].copy()
        # Start the line where the arm is *commanded* to be (as Executor.descend_until
        # does): under the can's load the measured wrist sags a few mm below it, and a
        # line starting there pulled the hand back down as the lift began -- the grip
        # went to 0N at that instant and the can slid out on the way up (pick
        # (0.28,-0.265), centre basket, 2026-09-25).
        start = scene.wrist_position_at(side, here)
        target = plan.centers["lift"].copy()
        target[:2] = start[:2]
        try:
            path = self.planner._walk(start, target, here, C.LIFT_PATH_STEPS)
        except RuntimeError:
            self.resolve_lift_from_here()
            ex.move_to({arm: plan["lift"]}, C.MOVE_TO_LIFT)
            return
        if (min(self.planner.joint_margin_degrees(q) for q in path) < C.MIN_JOINT_MARGIN_DEG
                or min(self.planner.arm_body_clearance(q) for q in path) < C.ARM_BODY_CLEARANCE):
            self.resolve_lift_from_here()
            ex.move_to({arm: plan["lift"]}, C.MOVE_TO_LIFT)
            return
        ex.follow({arm: path}, [C.MOVE_TO_LIFT / len(path)] * len(path))
        plan.joints["lift"] = path[-1]
        plan.centers["lift"] = target

    def place_floor(self) -> np.ndarray | None:
        return None if self.place_offset is None else self.scene.basket_floor() + self.place_offset

    def phase_plan(self) -> None:
        self.plan = self.executor.think(
            self.planner.plan, self.object_position(), exclude_yaws_deg=tuple(self.failed_grasp_yaws), place_floor=self.place_floor()
        )
        self.log.record("grasp_yaw_deg", float(self.plan.grasp_yaw_deg))
        self.log.record("place_yaw_deg", float(self.plan.place_yaw_deg))
        self.log.record("route_strategy", self.plan.route_strategy)
        self.log.record(
            "min_joint_margin_deg",
            min(self.planner.joint_margin_degrees(joints) for joints in self.plan.joints.values()),
        )
        self.log.record("phase_wrist_positions", {k: v.tolist() for k, v in self.plan.centers.items()})
        self.log.record("phase_joint_targets", {k: v.tolist() for k, v in self.plan.joints.items()})
        self.log.note(f"waypoints solved via {self.plan.route_strategy}, set-down hand yaw {self.plan.place_yaw_deg:+.0f} degrees")
        if self.log.verbose:
            print(self.planner.describe(self.plan))

    def phase_ready(self) -> None:
        ex, plan = self.executor, self.plan
        ex.hold(C.SETTLE_AT_START)
        arm = f"{self.side}_arm"
        ex.move_to({arm: plan["raise"]}, C.MOVE_TO_RAISE)
        ex.move_to({arm: plan["hover"]}, C.MOVE_TO_HOVER)
        ex.move_to({arm: plan["ready"]}, C.MOVE_TO_READY)

    def phase_reach(self) -> None:
        ex, plan, scene = self.executor, self.plan, self.scene
        # Open wide *before* travelling, slide in horizontally so the object enters
        # between the jaws, close only once the arm has arrived.
        side, arm = self.side, f"{self.side}_arm"
        ex.preshape_hand(side)
        ex.hold(C.PRESHAPE_SETTLE)
        ex.move_to({arm: plan["pregrasp"]}, C.MOVE_TO_PREGRASP)
        ex.hold(C.PREGRASP_SETTLE)
        touching = {n: f for n, f in scene.finger_contact_forces(side).items() if f > 0.05}
        if touching:
            raise RuntimeError(
                f"standoff pose is already touching the object ({touching}); "
                f"increase APPROACH_STANDOFF (currently {C.APPROACH_STANDOFF:.2f}m)"
            )
        ex.move_to({arm: plan["grasp"]}, C.MOVE_TO_GRASP)
        bend = [float(np.degrees(self.data.qpos[scene.arm_qpos[side][i]])) for i in C.WRIST_BEND_INDICES]
        self.log.record("wrist_pitch_at_grasp_deg", float(np.hypot(*bend)))
        self.log.note(f"wrist bend at grasp: joint6 {bend[0]:+.1f}, joint7 {bend[1]:+.1f} degrees (0 = hand in line with forearm)")

    def phase_grasp(self) -> None:
        ex, scene, side = self.executor, self.scene, self.side
        fingers = ("index", "middle", "ring", "pinky")
        ex.close_until_contact(side, fingers, force_target=1.0)
        ex.close_until_contact(side, ("thumb",), force_target=1.0)
        # Remember the light-contact hand pose: the release returns to it first.
        self.light_grip_ctrl = scene.data.ctrl[scene.hand_actuators[side]].copy()
        forces = ex.close_until_contact(side, (*fingers, "thumb"))
        self.log.note("contact force per finger (N): " + ", ".join(f"{k}={v:.2f}" for k, v in forces.items()))
        # Every finger must be on the object before the hand moves at all; a finger that
        # stopped short is closed further on its own.
        missing = self.fingers_not_pressing()
        if missing:
            forces = ex.close_until_contact(side, tuple(missing))
            missing = self.fingers_not_pressing()
        if missing:
            raise RuntimeError(f"not all fingers touch the object before lift; missing {missing}: {forces}")
        self.log.record("grasp_forces", scene.finger_contact_forces(side))
        self.log.note("thumb opposed by at least two fingers -> proof lift")

        grasp_orientation = self.plan.grasp_orientation if self.plan.grasp_orientation is not None else self.planner.orientation
        rise, tilt, hand_rise = ex.proof_lift(side, grasp_orientation)
        slip = hand_rise - rise
        self.log.record("proof_lift_rise_m", rise)
        self.log.record("proof_lift_hand_rise_m", hand_rise)
        self.log.record("proof_lift_tilt_deg", tilt)
        self.log.note(f"proof lift: hand {hand_rise*100:+.1f}cm, object {rise*100:+.1f}cm (slip {slip*1000:.0f}mm), tilt {tilt:.0f} degrees")
        if hand_rise < C.PROOF_LIFT_MIN_HAND_RISE:
            raise RuntimeError(f"proof lift did not happen: hand rose only {hand_rise*100:.1f}cm")
        if slip > C.PROOF_LIFT_SLIP_LIMIT[side] or tilt > C.PROOF_LIFT_TILT_LIMIT_DEG[side]:
            raise RuntimeError(
                f"grasp failed: object did not come with the hand (hand +{hand_rise*100:.1f}cm, "
                f"object +{rise*100:.1f}cm, slip {slip*1000:.0f}mm, tilt {tilt:.0f}deg); forces {forces}"
            )
        # Re-grip once the can hangs in the hand: its weight shifts it in the grasp
        # during the proof lift, and a finger that settled below the target force
        # (middle at 3-4N, default perception layout) let go when the arm stopped at
        # the top of the lift (2026-09-24). Fingers already on the can only; the
        # thumb is left alone (it already presses hardest, squeezing it turns the can).
        # Forces averaged over a short hold: the contacts chatter, and a single sample
        # taken the instant the proof lift stopped read ~0N on every finger (pick
        # (0.28,-0.265), 2026-09-25), so no finger was re-gripped -- the index at 3.9N
        # just before -- and the can slid out on the way to the basket.
        samples = []
        for _ in range(C.REGRIP_SAMPLES):
            ex.hold(C.REGRIP_SAMPLE_SECONDS / C.REGRIP_SAMPLES)
            samples.append(scene.finger_contact_forces(side))
        mean_force = {name: float(np.mean([s[name] for s in samples])) for name in samples[0]}
        on_can = tuple(
            name for name, force in mean_force.items()
            if 0.5 < force < C.REGRIP_BELOW_N[side] and name != "thumb"
        )
        if on_can:
            forces = ex.close_until_contact(side, on_can)
            self.log.note("re-grip after proof lift (N): " + ", ".join(f"{k}={v:.1f}" for k, v in forces.items()))

    def _check_carry_clearance(self, label: str) -> float:
        clearance = self.scene.object_bottom_z() - self.scene.basket_rim_z()
        if clearance < C.CARRY_CLEARANCE_ABOVE_RIM - 0.005:
            raise RuntimeError(
                f"{label} too low: object bottom only {clearance*100:.1f}cm above the rim, "
                f"needed {C.CARRY_CLEARANCE_ABOVE_RIM*100:.1f}cm"
            )
        return clearance

    def phase_carry(self) -> None:
        ex, plan, scene, side = self.executor, self.plan, self.scene, self.side
        arm = f"{side}_arm"
        from simulation.pick_place.kinematics import rotation_z, solve_pose_ik
        self.lift_straight_up()
        clearance = self._check_carry_clearance("lift")
        # Re-plan the set-down from where the object actually sits in the hand: the
        # fingers never close exactly on the nominal jaw centre, and that few-mm offset
        # otherwise turns into a placement error (2.5cm measured with the nominal jaw).
        held = scene.object_position() - scene.wrist_position(side)
        held_at_grasp_orientation = self.planner.orientation @ scene.wrist_rotation(side).T @ held
        try:
            self.executor.think(
                self.planner.plan_place, plan, self.object_position(), held_offset=held_at_grasp_orientation, place_floor=self.place_floor()
            )
            self.log.note(f"object held {np.round(held_at_grasp_orientation, 3).tolist()} from the wrist; set-down re-planned")
        except RuntimeError as error:
            # The nominal collision-checked set-down is still valid. A measured grip
            # offset can make the compensated pose unreachable near a workspace edge.
            self.log.note(f"held-offset compensation unavailable; using nominal set-down ({str(error).splitlines()[0]})")
        self.log.record("held_offset_m", held_at_grasp_orientation)
        self.log.record("carry_clearance_above_rim_m", clearance)
        self.log.note(f"object bottom is {clearance*100:+.1f}cm above the basket rim; transferring A -> B")
        path = plan.paths["transfer"]
        ex.follow({arm: path}, [C.TRANSFER_SECONDS / len(path)] * len(path))
        self._check_carry_clearance("transfer")
        # Re-centre at carry height, before entering the basket.  The old lower path
        # was solved from a stale held offset; using it after a lateral correction
        # made the left wrist enter the rim with no valid IK escape.
        if side == "left":
            error = scene.object_position()[:2] - scene.basket_floor()[:2]
            if float(np.linalg.norm(error)) > C.SET_DOWN_CENTRING_TOLERANCE:
                current = scene.wrist_position(side)
                target = current.copy()
                target[:2] -= error
                try:
                    seed = self.data.ctrl[scene.arm_actuators[side]].copy()
                    # A single large lateral move is often outside the mirrored
                    # left-arm IK basin. Try an intermediate approach and small
                    # approach-axis offsets before rejecting the placement.
                    delta = target - current
                    candidates = [
                        current + 0.5 * delta,
                        target,
                        target + np.array([0.04, 0.0, 0.0]),
                        target + np.array([-0.04, 0.0, 0.0]),
                        target + np.array([0.0, -0.04, 0.0]),
                        target + np.array([0.0, 0.04, 0.0]),
                    ]
                    chosen = None
                    yaw_candidates = [plan.place_yaw_deg, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
                    for yaw in dict.fromkeys(yaw_candidates):
                        orientation = rotation_z(yaw) @ self.planner.orientation
                        for candidate in candidates:
                            try:
                                q = solve_pose_ik(self.model, side, candidate, orientation, seed)
                                margin = self.planner.joint_margin_degrees(q)
                                if margin < C.MIN_JOINT_MARGIN_DEG:
                                    continue
                                if self.planner.hand_contacts(q, scene.basket_geoms, closed=True):
                                    continue
                                chosen = (candidate, q, yaw, orientation)
                                break
                            except RuntimeError:
                                continue
                        if chosen is not None:
                            break
                    if chosen is None:
                        raise RuntimeError("no intermediate left-arm centring pose passed IK/collision")
                    candidate, centre_joints, chosen_yaw, chosen_orientation = chosen
                    plan.place_yaw_deg = chosen_yaw
                    if not np.allclose(candidate, current):
                        ex.move_to({arm: centre_joints}, C.SET_DOWN_CENTRING_SECONDS)
                    # Refine residual error with small midpoint moves; this keeps each
                    # IK request inside the left-arm basin instead of demanding one
                    # unreachable lateral jump near the workspace edge.
                    for _ in range(2):
                        residual = scene.object_position()[:2] - scene.basket_floor()[:2]
                        if float(np.linalg.norm(residual)) <= C.SET_DOWN_CENTRING_TOLERANCE:
                            break
                        step_target = scene.wrist_position(side).copy()
                        step_target[:2] -= 0.5 * residual
                        try:
                            step_q = solve_pose_ik(
                                self.model, side, step_target, chosen_orientation,
                                self.data.ctrl[scene.arm_actuators[side]].copy(),
                            )
                            if self.planner.hand_contacts(step_q, scene.basket_geoms, closed=True):
                                break
                            ex.move_to({arm: step_q}, C.SET_DOWN_CENTRING_SECONDS)
                        except RuntimeError:
                            break
                    self.log.record("carry_centring_error_m", error)
                    self.log.note(f"left carry-height centring corrected {np.linalg.norm(error)*1000:.1f}mm")
                    # Rebuild the lower path from the actual post-centring joint state.
                    lower_target = scene.wrist_position(side).copy()
                    lower_target[2] = plan.centers["lower"][2]
                    lower_joints = solve_pose_ik(
                        self.model, side, lower_target,
                        chosen_orientation,
                        self.data.ctrl[scene.arm_actuators[side]].copy(),
                    )
                    lower_path = self.planner._walk(
                        scene.wrist_position(side), lower_target,
                        self.data.ctrl[scene.arm_actuators[side]].copy(), C.CARRY_PATH_STEPS,
                        orientation_at=lambda f, yaw=plan.place_yaw_deg:
                            rotation_z(yaw) @ self.planner.orientation,
                    )
                    plan.paths["lower"] = lower_path
                    plan.joints["lower"] = lower_joints
                except RuntimeError as error:
                    self.log.note(f"left carry-height centring rejected: {str(error).splitlines()[0]}")
        self.log.note("lowering the object into the basket")
        path = plan.paths["lower"]
        ex.follow({arm: path}, [C.LOWER_SECONDS / len(path)] * len(path))
        # Set the object down for real: keep descending until it rests on the floor.
        orientation = rotation_z(plan.place_yaw_deg) @ self.planner.orientation
        self._centre_over_basket(orientation)
        # Stop only once the object stands flat: a can held slightly tilted touches the
        # floor on its rim first, and releasing it there lets it topple onto its base
        # and skid a centimetre or more. A few more millimetres of descent let the
        # fingers give and the base settle before the hand opens.
        _, height = scene.object_extents()
        # basket_floor() is the floor plate's centre; the can stands on its top face.
        floor_half = float(self.model.geom("place_basket_bottom").size[2])
        resting_z = float(scene.basket_floor()[2]) + floor_half + 0.5 * height

        def seated() -> bool:
            # On a V-floor insert the can meets a plate a few mm above the floor.
            slack = 0.0 if not scene.basket_floor_tilt_deg else BASKET_HALF_WIDTH * np.tan(np.deg2rad(scene.basket_floor_tilt_deg))
            return scene.object_on_basket_floor() and float(scene.object_position()[2]) <= resting_z + C.SET_DOWN_SEATED_TOLERANCE + slack

        went = ex.descend_until(side, orientation, seated)
        touching = scene.object_on_basket_floor()
        self.log.record("set_down_descent_m", went)
        self.log.note(f"descended {went*100:.1f}cm more; object {'rests on' if touching else 'is NOT on'} the basket floor")
        if not touching:
            # The arm can run out of reach a few mm above the floor (at full extension
            # after the robot was re-measured 3cm lower, 2026-09-23: 4mm short on the
            # default layout). Letting go from that low is a benign drop; anything
            # higher, or with the can not fully over the basket floor, is a failure.
            _, height = scene.object_extents()
            gap = float(scene.object_position()[2]) - 0.5 * height - float(scene.basket_floor()[2]) - 0.005
            if gap > C.SET_DOWN_MAX_RELEASE_GAP or not scene.object_inside_basket():
                raise RuntimeError(f"set-down failed: object still off the floor after {went*100:.1f}cm of descent")
            self.log.record("set_down_release_gap_m", gap)
            self.log.note(f"arm at full reach; releasing {gap*1000:.1f}mm above the basket floor")

    def _centre_over_basket(self, orientation: np.ndarray) -> None:
        """Slide the wrist so the object -- not the wrist -- hangs over the basket centre.

        The set-down is planned from the grip offset measured back at the lift, but the
        object creeps in the hand during the carry, so by the time it is over the basket
        that offset no longer describes it: measured across 18 trials it came down a
        systematic 13mm short in both axes of where the wrist had aimed it. Closing the
        loop here, on the object's real position at the moment it matters, is the same
        idea as the lift-time re-plan -- just applied late enough to still be true.
        """
        from simulation.pick_place.kinematics import solve_pose_ik

        ex, scene, side = self.executor, self.scene, self.side
        error = scene.object_position()[:2] - scene.basket_floor()[:2]
        if float(np.linalg.norm(error)) <= C.SET_DOWN_CENTRING_TOLERANCE:
            return
        original_z = float(scene.wrist_position(side)[2])
        target = scene.wrist_position(side)
        target[:2] -= error
        # The mirrored left wrist often reaches the basket centre at floor height
        # outside its IK workspace or inside the rim.  Solve the lateral correction
        # from a raised hand, then let the normal seated descent close the final gap.
        if side == "left":
            target[2] += 0.03
        try:
            joints = solve_pose_ik(
                self.model, side, target, orientation, self.data.ctrl[scene.arm_actuators[side]].copy()
            )
        except RuntimeError as failure:
            self.log.note(f"centring over the basket unavailable ({failure}); setting down as planned")
            return
        # Near the floor the hand sits deep between the basket walls, where a sideways
        # correction can put a knuckle through one. Skip the correction rather than
        # abort the set-down over it.
        hits = self.planner.hand_contacts(joints, scene.basket_geoms, closed=True)
        if hits:
            self.log.note(f"centring would put {', '.join(sorted(hits))} into the basket; setting down as planned")
            return
        start_joints = self.data.ctrl[scene.arm_actuators[side]].copy()
        ex.move_to({f"{side}_arm": joints}, C.SET_DOWN_CENTRING_SECONDS)
        if side == "left":
            # Return to the original vertical level after the raised lateral move;
            # the normal seated descent then remains responsible for floor contact.
            lowered = target.copy()
            lowered[2] = original_z
            try:
                lowered_joints = solve_pose_ik(
                    self.model, side, lowered, orientation, self.data.ctrl[scene.arm_actuators[side]].copy()
                )
                ex.move_to({f"{side}_arm": lowered_joints}, C.SET_DOWN_CENTRING_SECONDS)
            except RuntimeError:
                self.log.note("left centring lowered pose unavailable; reverting to pre-centring pose")
                ex.move_to({f"{side}_arm": start_joints}, C.SET_DOWN_CENTRING_SECONDS)
        moved = scene.object_position()[:2] - scene.basket_floor()[:2]
        self.log.record("set_down_centring_m", float(np.linalg.norm(error)))
        self.log.note(
            f"object was {np.linalg.norm(error)*1000:.1f}mm off the basket centre; "
            f"centred to {np.linalg.norm(moved)*1000:.1f}mm before setting down"
        )

    def phase_release(self) -> None:
        ex, plan, scene, side = self.executor, self.plan, self.scene, self.side
        arm, hand = f"{side}_arm", f"{side}_hand"
        object_before = scene.object_position().copy()
        wrist_before = scene.wrist_position(side).copy()
        self.log.record("release_object_before_m", object_before)
        self.log.record("release_wrist_before_m", wrist_before)
        # The object already rests on the basket floor (phase_carry). Release in three
        # steps, measured across four cases (two layouts, with and without RGB-D;
        # placement 1-12mm, every other ordering tried was worse, up to toppling the
        # can): first relax the grip back to the light-contact pose so no finger keeps
        # pushing once its opposite lets go, then uncurl all five fingers together
        # while the wrist retreats straight up, and only at carry height swing the
        # thumb out of opposition -- that yaw swing sweeps sideways through the can's
        # footprint and shoved it 25mm when done at set-down height.
        ex.move_to({hand: self.light_grip_ctrl}, C.RELAX_GRIP_SECONDS)
        if side == "left":
            # The mirrored RH56 collision geometry releases cleanly by opening first;
            # coupling opening to the retreat sweeps its index side through the can.
            ex.open_fingers(side, C.ALL_FINGERS, C.RELEASE_SECONDS)
            object_after_open = scene.object_position().copy()
            ex.move_to({arm: plan["transfer"]}, C.RETREAT_SECONDS)
        else:
            ex.move_to({hand: scene.hand_ctrl(side, open_fingers=C.ALL_FINGERS), arm: plan["transfer"]}, C.RETREAT_SECONDS)
            object_after_open = scene.object_position().copy()
        object_after_retreat = scene.object_position().copy()
        wrist_after_retreat = scene.wrist_position(side).copy()
        self.log.record("release_object_after_open_m", object_after_open)
        self.log.record("release_object_after_retreat_m", object_after_retreat)
        self.log.record("release_wrist_after_retreat_m", wrist_after_retreat)
        self.log.record("release_open_displacement_m", float(np.linalg.norm(object_after_open - object_before)))
        self.log.record("release_retreat_displacement_m", float(np.linalg.norm(object_after_retreat - object_after_open)))
        self.log.record("release_lateral_drift_m", float(np.linalg.norm((object_after_retreat - object_before)[:2])))
        ex.open_fingers(side, ("thumb",), 0.5 * C.RELEASE_SECONDS, release_thumb_yaw=True)
        # Continue home after clearing the basket.
        ex.move_to({arm: plan["hover"], hand: scene.rest_hand[side]}, C.RETURN_SECONDS)
        # Back the way it came, via a raise point re-chosen now that the object stands
        # in the basket, so the hand never sweeps low over the basket or the object.
        try:
            raise_joints, _ = ex.think(self.planner.find_raise, plan["hover"], plan.centers["hover"])
        except RuntimeError:
            # No reachable raise point (left arm after a centre-basket retrieval,
            # real-height robot, 2026-09-24). Go straight home only if that blend is
            # clear of the basket, the object and the table.
            obstacles = scene.basket_geoms | {scene.object_geom} | scene.table_geoms
            blocked = self.planner.blend_contacts(plan["hover"], scene.attention_pose[side], obstacles)
            if blocked:
                raise
            self.log.note("no raise way point on the way back; the direct blend home is clear")
            raise_joints = None
        if raise_joints is not None:
            ex.move_to({arm: raise_joints}, C.RETURN_SECONDS)
        ex.move_to({arm: scene.attention_pose[side]}, C.RETURN_SECONDS)
        ex.hold(C.FINAL_SETTLE)  # settle to verify the object stands on its own

    PHASE_MESSAGES = {
        "perceive": "locating the object with the head camera",
        "plan": "deriving wrist targets from object, basket and hand geometry, solving IK chain",
        "ready": "attention stance, then raising the selected arm to hover",
        "reach": "opening the hand, sliding in horizontally around the object",
        "grasp": "making light opposing contact, increasing grip force, proof lift",
        "carry": "lifting above the basket rim, moving A -> B, lowering",
        "release": "opening the hand, setting the object down, retreating",
    }

    def run(self, viewer=None, stop_after: str | None = None) -> None:
        """Run the episode. Raises RuntimeError from the phase that failed; the log's
        `current_phase` says which."""
        if stop_after is not None and stop_after not in PHASES:
            raise ValueError(f"unknown phase {stop_after!r}; choose from {PHASES}")
        self.executor.viewer = viewer
        index = 0
        while index < len(PHASES):
            name = PHASES[index]
            self.log.phase(index + 1, len(PHASES), name, self.PHASE_MESSAGES[name])
            try:
                getattr(self, f"phase_{name}")()
            except RuntimeError as error:
                # A grasp the physics rejects (a finger not pressing, the proof lift
                # leaving the object behind) is retried with the next grasp heading:
                # IK feasibility alone does not tell which heading holds the can.
                if name != "grasp" or len(self.failed_grasp_yaws) >= C.GRASP_RETRIES:
                    raise
                self.failed_grasp_yaws.append(float(self.plan.grasp_key))
                self.log.note(f"grasp at yaw {self.plan.grasp_yaw_deg:+.0f} rejected ({error}); letting go and retrying")
                self.recover_from_failed_grasp()
                index = PHASES.index("perceive")
                continue
            self.log.observe(name, object=self.scene.object_position(), wrist=self.scene.wrist_position(self.side))
            if name == stop_after:
                self.log.note(f"stopped after '{name}' as requested")
                return
            index += 1

    def recover_from_failed_grasp(self) -> None:
        """Let go of the object where it is and return to attention, the same way the
        release does, so the episode can plan afresh from a clean stance."""
        ex, plan, scene, side = self.executor, self.plan, self.scene, self.side
        arm, hand = f"{side}_arm", f"{side}_hand"
        if self.light_grip_ctrl is not None:
            ex.move_to({hand: self.light_grip_ctrl}, C.RELAX_GRIP_SECONDS)
        ex.move_to({hand: scene.hand_ctrl(side, open_fingers=C.ALL_FINGERS), arm: plan["hover"]}, C.RETREAT_SECONDS)
        ex.open_fingers(side, ("thumb",), 0.5 * C.RELEASE_SECONDS, release_thumb_yaw=True)
        ex.move_to({hand: scene.rest_hand[side]}, 0.5 * C.RELEASE_SECONDS)
        raise_joints, _ = ex.think(self.planner.find_raise, plan["hover"], plan.centers["hover"])
        ex.move_to({arm: raise_joints}, C.RETURN_SECONDS)
        ex.move_to({arm: scene.attention_pose[side]}, C.RETURN_SECONDS)
        ex.hold(C.SETTLE_AT_START)
        self.perceived_position = None  # look again: the failed grasp may have moved it

    def restart(self) -> None:
        """Put the scene back at the start of an episode and clear the log, so `run()`
        can be called again on the same model/data (the viewer stays attached)."""
        self.scene.reset()
        self.log = EpisodeLog(lambda: self.scene.data.time, verbose=self.log.verbose)
        self.executor = Executor(self.scene, on_step=self.executor.on_step)
        self.executor.active_side = self.side
        self.perceived_position = None
        self.light_grip_ctrl: np.ndarray | None = None
        self.failed_grasp_yaws: list[float] = []
        self.perception_error_m = None
        self.plan = None

    # ------------------------------------------------------------------ compatibility
    # Older call sites and tests used these private names; keep them as thin aliases.
    def _right_grasp_is_secure(self) -> bool:
        return self.grasp_is_secure()

    def _solve_poses(self) -> dict[str, dict[str, np.ndarray]]:
        self.plan = self.planner.plan(self.object_position())
        return {self.side: {**self.plan.joints, **{f"{k}_path": v for k, v in self.plan.paths.items()}}}


# ---------------------------------------------------------------------- trials / layouts
def object_to_basket_distance(pick, basket, half_width: float = BASKET_HALF_WIDTH + BASKET_WALL_THICKNESS) -> float:
    """Table-plane distance from the object's centre to the basket's nearest outer wall."""
    dx = max(abs(pick[0] - basket[0]) - half_width, 0.0)
    dy = max(abs(pick[1] - basket[1]) - half_width, 0.0)
    return float(np.hypot(dx, dy))


def sample_layout(
    rng: np.random.Generator, *, side: str = "right", perception: bool = False, attempts: int = 50
) -> dict:
    """Draw an object/basket layout the planner can actually execute.

    Randomization must preserve the success condition: a sample is accepted only once
    the whole waypoint chain solves on a throwaway scene, so a rejected sample costs a
    few seconds of planning rather than a physics trial that says nothing about the grasp.
    """
    for _ in range(attempts):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        mirror = 1.0 if side == "right" else -1.0
        pick = (float(rng.uniform(*C.RANDOM_PICK_BOX[0])), mirror * float(rng.uniform(*C.RANDOM_PICK_BOX[1])))
        basket = (float(rng.uniform(*C.RANDOM_BASKET_BOX[0])), mirror * float(rng.uniform(*C.RANDOM_BASKET_BOX[1])))
        if np.hypot(basket[0] - pick[0], basket[1] - pick[1]) < C.MIN_PICK_TO_BASKET_M:
            continue
        if object_to_basket_distance(pick, basket) < C.MIN_OBJECT_TO_BASKET_M:
            continue
        try:
            Demo(pick, basket, perception=perception, side=side)._solve_poses()
        except (RuntimeError, ValueError):
            continue
        return dict(pick_position=pick, basket_position=basket)
    raise RuntimeError(f"no executable layout found in {attempts} samples")


def run_trial(demo: Demo, viewer=None, stop_after: str | None = None) -> TrialResult:
    """Run one episode to completion (or failure) and record everything about it."""
    failure = None
    try:
        demo.run(viewer, stop_after=stop_after)
    except RuntimeError as error:
        failure = str(error)
    scene, values = demo.scene, demo.log.values
    final_pos = scene.object_position()
    placement_error = float(np.linalg.norm(final_pos[:2] - scene.basket_floor()[:2]))
    inside_basket = scene.object_inside_basket()
    tilt = upright_tilt_degrees(scene.object_quaternion())
    if failure is None and stop_after is None:
        if not inside_basket:
            failure = f"object footprint is outside basket (placement error {placement_error*1000:.1f}mm)"
        elif tilt > C.PROOF_LIFT_MAX_TILT_DEG:
            failure = f"final object tilt {tilt:.1f}deg exceeds {C.PROOF_LIFT_MAX_TILT_DEG:.0f}deg"
    return TrialResult(
        failure is None,
        failure,
        demo.log.current_phase if failure else None,
        final_pos.tolist(),
        float(scene.data.time),
        placement_error_m=placement_error,
        inside_basket=inside_basket,
        bottle_tilt_deg=tilt,
        contact_forces=scene.finger_contact_forces(demo.side),
        pick_position=list(scene.pick_position),
        basket_position=list(scene.basket_position),
        perception_used=demo.perception,
        perceived_position=values.get("perceived_position"),
        perception_error_m=values.get("perception_error_m"),
        grasp_forces=values.get("grasp_forces"),
        wrist_pitch_at_grasp_deg=values.get("wrist_pitch_at_grasp_deg"),
        proof_lift_rise_m=values.get("proof_lift_rise_m"),
        proof_lift_hand_rise_m=values.get("proof_lift_hand_rise_m"),
        proof_lift_tilt_deg=values.get("proof_lift_tilt_deg"),
        carry_clearance_above_rim_m=values.get("carry_clearance_above_rim_m"),
        place_yaw_deg=values.get("place_yaw_deg"),
        release_object_before_m=values.get("release_object_before_m"),
        release_object_after_open_m=values.get("release_object_after_open_m"),
        release_object_after_retreat_m=values.get("release_object_after_retreat_m"),
        release_wrist_before_m=values.get("release_wrist_before_m"),
        release_wrist_after_retreat_m=values.get("release_wrist_after_retreat_m"),
        release_open_displacement_m=values.get("release_open_displacement_m"),
        release_retreat_displacement_m=values.get("release_retreat_displacement_m"),
        release_lateral_drift_m=values.get("release_lateral_drift_m"),
        phase_wrist_positions=values.get("phase_wrist_positions"),
        phase_joint_targets=values.get("phase_joint_targets"),
        phase_observations=demo.log.observations,
        route=f"DIRECT_{demo.side.upper()}",
        route_reason=demo.route_reason,
        source_arm=demo.side,
        target_arm=demo.side,
        min_joint_margin_deg=values.get("min_joint_margin_deg"),
        max_penetration_m=demo.executor.max_penetration_m,
        model_timestep_s=float(scene.model.opt.timestep),
    )
