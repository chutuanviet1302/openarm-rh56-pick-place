"""The pick-and-place episode as an explicit sequence of phases.

    Demo(...)            builds Scene + GraspPlanner + Executor + EpisodeLog
    demo.run(viewer)     perceive -> plan -> ready -> reach -> grasp -> carry -> release
    demo.run(stop_after="grasp")   stop with the scene left in that phase's end state

Each `phase_*` method does one thing, checks one thing, and records its evidence in
`demo.log.values`; `run_trial()` turns that into a TrialResult.
"""

from __future__ import annotations

import numpy as np

from simulation.five_finger_model import BASKET_POSITION_B, PICK_POSITION_A
from simulation.pick_place import config as C
from simulation.pick_place.episode import EpisodeLog, TrialResult
from simulation.pick_place.executor import Executor
from simulation.pick_place.kinematics import upright_tilt_degrees
from simulation.pick_place.planner import PHASE_ORDER, GraspPlanner, Plan
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
    ) -> None:
        self.scene = Scene(pick_position, basket_position)
        self.planner = GraspPlanner(self.scene)
        self.executor = Executor(self.scene)
        self.log = EpisodeLog(lambda: self.scene.data.time, verbose=verbose)
        # With perception on, the object's position comes from the camera (RGB-D ->
        # deprojection), never from the simulator state; the error against ground truth
        # is logged so a calibration or segmentation fault is visible in the report.
        self.perception = perception
        self.perception_camera = perception_camera
        self.perceived_position: np.ndarray | None = None
        self.perception_error_m: float | None = None
        self.plan: Plan | None = None

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
        forces = self.scene.finger_contact_forces("right")
        required = C.FINGER_NAMES if self.REQUIRE_THUMB_OPPOSITION else tuple(n for n in C.FINGER_NAMES if n != "thumb")
        return [name for name in required if forces[name] < C.GRASP_SECURE_MIN_FORCE_N]

    def grasp_is_secure(self) -> bool:
        """Force-based: every finger must press on the object before the lift is allowed."""
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

    def phase_plan(self) -> None:
        self.plan = self.planner.plan(self.object_position())
        self.log.record("place_yaw_deg", float(self.plan.place_yaw_deg))
        self.log.record("phase_wrist_positions", {k: v.tolist() for k, v in self.plan.centers.items()})
        self.log.record("phase_joint_targets", {k: v.tolist() for k, v in self.plan.joints.items()})
        self.log.note(f"waypoints solved, set-down hand yaw {self.plan.place_yaw_deg:+.0f} degrees")
        if self.log.verbose:
            print(self.planner.describe(self.plan))

    def phase_ready(self) -> None:
        ex, plan = self.executor, self.plan
        ex.hold(C.SETTLE_AT_START)
        ex.move_to({"right_arm": plan["hover"]}, C.MOVE_TO_HOVER)
        ex.move_to({"right_arm": plan["ready"]}, C.MOVE_TO_READY)

    def phase_reach(self) -> None:
        ex, plan, scene = self.executor, self.plan, self.scene
        # Open wide *before* travelling, slide in horizontally so the object enters
        # between the jaws, close only once the arm has arrived.
        ex.preshape_hand("right")
        ex.hold(C.PRESHAPE_SETTLE)
        ex.move_to({"right_arm": plan["pregrasp"]}, C.MOVE_TO_PREGRASP)
        ex.hold(C.PREGRASP_SETTLE)
        touching = {n: f for n, f in scene.finger_contact_forces("right").items() if f > 0.05}
        if touching:
            raise RuntimeError(
                f"standoff pose is already touching the object ({touching}); "
                f"increase APPROACH_STANDOFF (currently {C.APPROACH_STANDOFF:.2f}m)"
            )
        ex.move_to({"right_arm": plan["grasp"]}, C.MOVE_TO_GRASP)
        bend = [float(np.degrees(self.data.qpos[scene.arm_qpos["right"][i]])) for i in C.WRIST_BEND_INDICES]
        self.log.record("wrist_pitch_at_grasp_deg", float(np.hypot(*bend)))
        self.log.note(f"wrist bend at grasp: joint6 {bend[0]:+.1f}, joint7 {bend[1]:+.1f} degrees (0 = hand in line with forearm)")

    def phase_grasp(self) -> None:
        ex, scene = self.executor, self.scene
        ex.close_until_contact("right", ("index", "middle", "ring", "pinky"))
        forces = ex.close_until_contact("right", ("thumb",))
        self.log.note("contact force per finger (N): " + ", ".join(f"{k}={v:.2f}" for k, v in forces.items()))
        # Every finger must be on the object before the hand moves at all; a finger that
        # stopped short is closed further on its own.
        missing = self.fingers_not_pressing()
        if missing:
            forces = ex.close_until_contact("right", tuple(missing))
            missing = self.fingers_not_pressing()
        if missing:
            raise RuntimeError(f"not all fingers touch the object before lift; missing {missing}: {forces}")
        self.log.record("grasp_forces", scene.finger_contact_forces("right"))
        self.log.note("all 5 fingers in contact -> proof lift")

        rise, tilt, hand_rise = ex.proof_lift()
        slip = hand_rise - rise
        self.log.record("proof_lift_rise_m", rise)
        self.log.record("proof_lift_hand_rise_m", hand_rise)
        self.log.record("proof_lift_tilt_deg", tilt)
        self.log.note(f"proof lift: hand {hand_rise*100:+.1f}cm, object {rise*100:+.1f}cm (slip {slip*1000:.0f}mm), tilt {tilt:.0f} degrees")
        if hand_rise < C.PROOF_LIFT_MIN_HAND_RISE:
            raise RuntimeError(f"proof lift did not happen: hand rose only {hand_rise*100:.1f}cm")
        if slip > C.PROOF_LIFT_MAX_SLIP or tilt > C.PROOF_LIFT_MAX_TILT_DEG:
            raise RuntimeError(
                f"grasp failed: object did not come with the hand (hand +{hand_rise*100:.1f}cm, "
                f"object +{rise*100:.1f}cm, slip {slip*1000:.0f}mm, tilt {tilt:.0f}deg); forces {forces}"
            )

    def _check_carry_clearance(self, label: str) -> float:
        clearance = self.scene.object_bottom_z() - self.scene.basket_rim_z()
        if clearance < C.CARRY_CLEARANCE_ABOVE_RIM - 0.005:
            raise RuntimeError(
                f"{label} too low: object bottom only {clearance*100:.1f}cm above the rim, "
                f"needed {C.CARRY_CLEARANCE_ABOVE_RIM*100:.1f}cm"
            )
        return clearance

    def phase_carry(self) -> None:
        ex, plan, scene = self.executor, self.plan, self.scene
        ex.move_to({"right_arm": plan["lift"]}, C.MOVE_TO_LIFT)
        clearance = self._check_carry_clearance("lift")
        # Re-plan the set-down from where the object actually sits in the hand: the
        # fingers never close exactly on the nominal jaw centre, and that few-mm offset
        # otherwise turns into a placement error (2.5cm measured with the nominal jaw).
        held = scene.object_position() - scene.wrist_position("right")
        held_at_grasp_orientation = scene.grasp_orientation @ scene.wrist_rotation("right").T @ held
        self.planner.plan_place(plan, self.object_position(), held_offset=held_at_grasp_orientation)
        self.log.record("held_offset_m", held_at_grasp_orientation)
        self.log.note(f"object held {np.round(held_at_grasp_orientation, 3).tolist()} from the wrist; set-down re-planned")
        self.log.record("carry_clearance_above_rim_m", clearance)
        self.log.note(f"object bottom is {clearance*100:+.1f}cm above the basket rim; transferring A -> B")
        path = plan.paths["transfer"]
        ex.follow({"right_arm": path}, [C.TRANSFER_SECONDS / len(path)] * len(path))
        self._check_carry_clearance("transfer")
        self.log.note("lowering the object into the basket")
        path = plan.paths["lower"]
        ex.follow({"right_arm": path}, [C.LOWER_SECONDS / len(path)] * len(path))
        # Set the object down for real: keep descending until it rests on the floor.
        from simulation.pick_place.kinematics import rotation_z

        orientation = rotation_z(plan.place_yaw_deg) @ scene.grasp_orientation
        went = ex.descend_until("right", orientation, lambda: scene.object_touches("place_basket_bottom"))
        touching = scene.object_touches("place_basket_bottom")
        self.log.record("set_down_descent_m", went)
        self.log.note(f"descended {went*100:.1f}cm more; object {'rests on' if touching else 'is NOT on'} the basket floor")
        if not touching:
            raise RuntimeError(f"set-down failed: object still off the floor after {went*100:.1f}cm of descent")

    def phase_release(self) -> None:
        ex, plan, scene = self.executor, self.plan, self.scene
        # The object already rests on the floor (phase_carry), so opening cannot drop it;
        # fingers first, then the thumb swings out of opposition.
        ex.open_fingers("right", ("index", "middle", "ring", "pinky"), 0.5 * C.RELEASE_SECONDS)
        ex.open_fingers("right", ("thumb",), 0.5 * C.RELEASE_SECONDS, release_thumb_yaw=True)
        ex.move_to({"right_hand": scene.open_hand["right"]}, 0.3)
        # Vertical retreat straight up out of the basket, then home.
        ex.move_to({"right_arm": plan["transfer"]}, C.RETREAT_SECONDS)
        ex.move_to({"right_arm": plan["hover"], "right_hand": scene.closed_hand["right"]}, C.RETURN_SECONDS)
        ex.move_to({"right_arm": scene.attention_pose["right"]}, C.RETURN_SECONDS)
        ex.hold(C.FINAL_SETTLE)  # settle to verify the object stands on its own

    PHASE_MESSAGES = {
        "perceive": "locating the object with the head camera",
        "plan": "deriving wrist targets from object, basket and hand geometry, solving IK chain",
        "ready": "attention stance, then raising the right arm to hover",
        "reach": "opening the hand, sliding in horizontally around the object",
        "grasp": "closing 4 fingers then thumb, verifying all 5 press, proof lift",
        "carry": "lifting above the basket rim, moving A -> B, lowering",
        "release": "opening the hand, setting the object down, retreating",
    }

    def run(self, viewer=None, stop_after: str | None = None) -> None:
        """Run the episode. Raises RuntimeError from the phase that failed; the log's
        `current_phase` says which."""
        if stop_after is not None and stop_after not in PHASES:
            raise ValueError(f"unknown phase {stop_after!r}; choose from {PHASES}")
        self.executor.viewer = viewer
        for index, name in enumerate(PHASES, start=1):
            self.log.phase(index, len(PHASES), name, self.PHASE_MESSAGES[name])
            getattr(self, f"phase_{name}")()
            self.log.observe(name, object=self.scene.object_position(), wrist=self.scene.wrist_position("right"))
            if name == stop_after:
                self.log.note(f"stopped after '{name}' as requested")
                return

    def restart(self) -> None:
        """Put the scene back at the start of an episode and clear the log, so `run()`
        can be called again on the same model/data (the viewer stays attached)."""
        self.scene.reset()
        self.log = EpisodeLog(lambda: self.scene.data.time, verbose=self.log.verbose)
        self.executor = Executor(self.scene, on_step=self.executor.on_step)
        self.perceived_position = None
        self.perception_error_m = None
        self.plan = None

    # ------------------------------------------------------------------ compatibility
    # Older call sites and tests used these private names; keep them as thin aliases.
    def _right_grasp_is_secure(self) -> bool:
        return self.grasp_is_secure()

    def _fingers_not_pressing(self) -> list[str]:
        return self.fingers_not_pressing()

    def _finger_contact_forces(self, side: str) -> dict[str, float]:
        return self.scene.finger_contact_forces(side)

    def _solve_poses(self) -> dict[str, dict[str, np.ndarray]]:
        self.plan = self.planner.plan(self.object_position())
        return {"right": {**self.plan.joints, **{f"{k}_path": v for k, v in self.plan.paths.items()}}}

    def _phase_centers(self, place_yaw_deg: float = 0.0) -> dict[str, np.ndarray]:
        return self.planner.centers(self.object_position(), place_yaw_deg)


# ---------------------------------------------------------------------- trials / layouts
def object_to_basket_distance(pick, basket, half_width: float = 0.085) -> float:
    """Table-plane distance from the object's centre to the basket's nearest outer wall."""
    dx = max(abs(pick[0] - basket[0]) - half_width, 0.0)
    dy = max(abs(pick[1] - basket[1]) - half_width, 0.0)
    return float(np.hypot(dx, dy))


def sample_layout(rng: np.random.Generator, *, perception: bool = False, attempts: int = 50) -> dict:
    """Draw an object/basket layout the planner can actually execute.

    Randomization must preserve the success condition: a sample is accepted only once
    the whole waypoint chain solves on a throwaway scene, so a rejected sample costs a
    few seconds of planning rather than a physics trial that says nothing about the grasp.
    """
    for _ in range(attempts):
        pick = (float(rng.uniform(*C.RANDOM_PICK_BOX[0])), float(rng.uniform(*C.RANDOM_PICK_BOX[1])))
        basket = (float(rng.uniform(*C.RANDOM_BASKET_BOX[0])), float(rng.uniform(*C.RANDOM_BASKET_BOX[1])))
        if np.hypot(basket[0] - pick[0], basket[1] - pick[1]) < C.MIN_PICK_TO_BASKET_M:
            continue
        if object_to_basket_distance(pick, basket) < C.MIN_OBJECT_TO_BASKET_M:
            continue
        try:
            Demo(pick, basket, perception=perception)._solve_poses()
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
    return TrialResult(
        failure is None,
        failure,
        demo.log.current_phase if failure else None,
        final_pos.tolist(),
        float(scene.data.time),
        placement_error_m=placement_error,
        bottle_tilt_deg=upright_tilt_degrees(scene.object_quaternion()),
        contact_forces=scene.finger_contact_forces("right"),
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
        phase_wrist_positions=values.get("phase_wrist_positions"),
        phase_joint_targets=values.get("phase_joint_targets"),
        phase_observations=demo.log.observations,
    )
