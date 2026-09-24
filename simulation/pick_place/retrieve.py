"""Place an object into the scene's one basket, then grasp it back out and set it
down at a different bare-table point.

Both legs run in the *same* MjModel/MjData (`Demo(..., scene=...)` continues the
physics instead of rebuilding a table with no basket at that spot), so the second
grasp really does reach in past the basket's walls for an object that is really
resting on its floor -- not a bare-table pick relabelled as a retrieval.

    RetrieveDemo(pick, basket, retrieve_to, side="left").run()
    run_retrieve_trial(...)   -> one TrialResult per leg, combined
"""

from __future__ import annotations

import numpy as np

from simulation.five_finger_model import TABLE_TOP_Z
from simulation.pick_place import config as C
from simulation.pick_place.demo import Demo, run_trial
from simulation.pick_place.episode import TrialResult
from simulation.pick_place.kinematics import rotation_z, upright_tilt_degrees
from simulation.pick_place.scene import Scene

RETRIEVE_PHASES = ("perceive", "plan", "ready", "reach", "grasp", "carry", "release")


def _table_floor(xy: tuple[float, float], surface_z: float = 0.0) -> np.ndarray:
    return np.array([float(xy[0]), float(xy[1]), float(TABLE_TOP_Z) + surface_z])


def _object_axis(scene: Scene) -> np.ndarray:
    import mujoco

    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, scene.object_quaternion())
    return matrix.reshape(3, 3)[:, 2]


def _rotation_onto_vertical(axis: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector `axis` (can axis, either sense) to +z."""
    a = axis / np.linalg.norm(axis)
    if a[2] < 0.0:
        a = -a
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z)
    s, c = float(np.linalg.norm(v)), float(a @ z)
    if s < 1e-9:
        return np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1.0 - c) / s**2)


def _partial_rotation(rotation: np.ndarray, fraction: float) -> np.ndarray:
    """`rotation` scaled to `fraction` of its angle (same axis)."""
    if fraction <= 0.0:
        return np.eye(3)
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return np.eye(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    axis /= np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    a = angle * fraction
    return np.eye(3) + np.sin(a) * k + (1.0 - np.cos(a)) * k @ k


def phase_plan(demo: Demo, retrieve_to: tuple[float, float]) -> None:
    """Like Demo.phase_plan, but the set-down target is the bare table point
    `retrieve_to` instead of the scene's one basket (which is where the object
    already sits -- that's what makes this a retrieval, not a fresh pick)."""
    demo.plan = demo.executor.think(
        demo.planner.plan, demo.object_position(), exclude_yaws_deg=tuple(demo.failed_grasp_yaws), place_floor=_table_floor(retrieve_to, demo.scene.work_surface_z)
    )
    demo.log.record("grasp_yaw_deg", float(demo.plan.grasp_yaw_deg))
    demo.log.record("place_yaw_deg", float(demo.plan.place_yaw_deg))
    demo.log.record("route_strategy", demo.plan.route_strategy)
    demo.log.note(
        f"retrieve plan: grasp yaw {demo.plan.grasp_yaw_deg:+.0f}, lift clear of the basket rim, "
        f"set down at {tuple(round(v, 3) for v in retrieve_to)} (yaw {demo.plan.place_yaw_deg:+.0f})"
    )


def phase_carry_out(demo: Demo, retrieve_to: tuple[float, float]) -> None:
    """Lift clear of the basket rim (the object still started inside it, so this
    check is exactly Demo._check_carry_clearance's), carry to `retrieve_to` and set
    down on the bare table -- Demo.phase_carry's basket-centring logic does not
    apply, there are no walls to fall short of here."""
    ex, plan, scene, side = demo.executor, demo.plan, demo.scene, demo.side
    arm = f"{side}_arm"
    demo.lift_straight_up()
    clearance = scene.object_bottom_z() - scene.basket_rim_z()
    if clearance < C.CARRY_CLEARANCE_ABOVE_RIM - 0.005:
        raise RuntimeError(
            f"lift too low: object bottom only {clearance*100:.1f}cm above the basket rim, "
            f"needed {C.CARRY_CLEARANCE_ABOVE_RIM*100:.1f}cm"
        )
    held = scene.object_position() - scene.wrist_position(side)
    planned_orientation = demo.planner.orientation
    # Level the can for the set-down. Picked off a sloped basket floor it sits in
    # the hand tilted (13deg measured, 2026-09-24); set down like that only one rim
    # edge meets the table and the can topples the moment the fingers open. The
    # hand is turned by the smallest rotation that brings the can's axis vertical.
    wrist_rotation = scene.wrist_rotation(side)
    axis_in_hand = wrist_rotation.T @ _object_axis(scene)
    level = _rotation_onto_vertical(planned_orientation @ axis_in_hand)
    tilt_deg = float(upright_tilt_degrees(scene.object_quaternion()))
    # Full levelling can sit outside the arm's reach at the set-down; take the
    # largest fraction of the correction the planner still solves.
    fractions = C.LEVEL_FRACTIONS if tilt_deg > C.LEVEL_BEFORE_SET_DOWN_DEG else ()
    replanned = False
    for fraction in (*fractions, 0.0):
        demo.planner.orientation = _partial_rotation(level, fraction) @ planned_orientation
        held_at_grasp_orientation = demo.planner.orientation @ wrist_rotation.T @ held
        try:
            demo.executor.think(
                demo.planner.plan_place, plan, demo.object_position(), held_offset=held_at_grasp_orientation, place_floor=_table_floor(retrieve_to, demo.scene.work_surface_z)
            )
        except RuntimeError as error:
            last_error = error
            continue
        replanned = True
        demo.log.note(f"object held {np.round(held_at_grasp_orientation, 3).tolist()} from the wrist; set-down re-planned")
        if fraction:
            demo.log.note(f"can tilted {tilt_deg:.0f}deg in the hand; hand turned to remove {fraction*100:.0f}% of it before set-down")
        break
    if not replanned:
        demo.planner.orientation = planned_orientation
        demo.log.note(f"held-offset compensation unavailable; using nominal set-down ({str(last_error).splitlines()[0]})")
    demo.log.record("carry_clearance_above_rim_m", clearance)
    demo.log.note(f"object bottom is {clearance*100:+.1f}cm above the basket rim; carrying out to {retrieve_to}")
    path = plan.paths["transfer"]
    ex.follow({arm: path}, [C.TRANSFER_SECONDS / len(path)] * len(path))
    demo.log.note("lowering the object onto the bare table")
    path = plan.paths["lower"]
    ex.follow({arm: path}, [C.LOWER_SECONDS / len(path)] * len(path))
    orientation = rotation_z(plan.place_yaw_deg) @ demo.planner.orientation
    _, height = scene.object_extents()
    resting_z = float(TABLE_TOP_Z) + scene.work_surface_z + 0.5 * height

    def seated() -> bool:
        return scene.object_on_work_surface() and float(scene.object_position()[2]) <= resting_z + C.SET_DOWN_SEATED_TOLERANCE

    went = ex.descend_until(side, orientation, seated)
    touching = scene.object_on_work_surface()
    demo.log.record("set_down_descent_m", went)
    demo.log.note(f"descended {went*100:.1f}cm more; object {'rests on' if touching else 'is NOT on'} the table")
    if not touching:
        raise RuntimeError(f"set-down failed: object still off the table after {went*100:.1f}cm of descent")


def run_retrieve(demo: Demo, retrieve_to: tuple[float, float], viewer=None, stop_after: str | None = None) -> None:
    """Run perceive -> plan -> ready -> reach -> grasp -> carry -> release on
    `demo`, whose scene already has the object resting in its one basket. The
    logged phase name is "carry" (matching Demo's, even though the work is
    `phase_carry_out`) so a shared timeline/viewer needs no special case for it.

    IK feasibility for a heading (checked at plan time) does not say whether that
    heading actually holds the can once fingers, walls and contact forces are real;
    a grasp physics rejects is retried with the next heading, exactly as Demo.run
    does for a direct pick -- the basket's walls rule out most headings up front, so
    retrieval leans on this retry more than a direct pick usually needs to.
    """
    demo.executor.viewer = viewer
    steps = (
        ("perceive", demo.phase_perceive),
        ("plan", lambda: phase_plan(demo, retrieve_to)),
        ("ready", demo.phase_ready),
        ("reach", demo.phase_reach),
        ("grasp", demo.phase_grasp),
        ("carry", lambda: phase_carry_out(demo, retrieve_to)),
        ("release", demo.phase_release),
    )
    index = 0
    while index < len(steps):
        name, fn = steps[index]
        demo.log.phase(index + 1, len(steps), name, name)
        try:
            fn()
        except RuntimeError as error:
            if name != "grasp" or len(demo.failed_grasp_yaws) >= C.GRASP_RETRIES:
                raise
            demo.failed_grasp_yaws.append(float(demo.plan.grasp_key))
            demo.log.note(f"grasp at yaw {demo.plan.grasp_yaw_deg:+.0f} rejected ({error}); letting go and retrying")
            demo.recover_from_failed_grasp()
            index = 0
            continue
        demo.log.observe(name, object=demo.scene.object_position(), wrist=demo.scene.wrist_position(demo.side))
        if name == stop_after:
            demo.log.note(f"stopped after '{name}' as requested")
            return
        index += 1


class RetrieveDemo:
    """`place_in`: an ordinary single-arm Demo (arm `place_side`) that puts the
    object in the basket at `basket_position`. `retrieve`: built only once
    `place_in` has actually run, from its live scene, so the grasp -- by arm
    `side`, which may be the other one -- is against the object where it really is.
    `place_side` defaults to `side` (one arm does both legs)."""

    def __init__(
        self,
        pick_position: tuple[float, float],
        basket_position: tuple[float, float],
        retrieve_to: tuple[float, float],
        *,
        side: str = "left",
        place_side: str | None = None,
        left_arm_mount_yaw_deg: float | None = None,
        right_arm_mount_yaw_deg: float | None = None,
        arm_half_separation: float | None = None,
        attention_deg: dict | None = None,
        place_offset: tuple[float, float] | None = None,
        basket_stand_height: float = 0.0,
        basket_floor_tilt_deg: float = 0.0,
        work_platform_height: float = 0.0,
        perception: bool = False,
        verbose: bool = False,
    ) -> None:
        self.pick_position = tuple(float(v) for v in pick_position)
        self.basket_position = tuple(float(v) for v in basket_position)
        self.retrieve_to = tuple(float(v) for v in retrieve_to)
        self.side = side
        self.place_side = place_side or side
        self.left_arm_mount_yaw_deg = left_arm_mount_yaw_deg
        self.perception = perception
        self.verbose = verbose
        scene = Scene(
            self.pick_position, self.basket_position,
            arm_half_separation=arm_half_separation, left_arm_mount_yaw_deg=left_arm_mount_yaw_deg,
            right_arm_mount_yaw_deg=right_arm_mount_yaw_deg, attention_deg=attention_deg,
            basket_stand_height=basket_stand_height, basket_floor_tilt_deg=basket_floor_tilt_deg,
            work_platform_height=work_platform_height,
        )
        self.place_in = Demo(perception=perception, verbose=verbose, side=self.place_side, scene=scene, place_offset=place_offset)
        self.retrieve: Demo | None = None

    def run(self, viewer=None, stop_after: str | None = None) -> None:
        self.place_in.run(viewer)  # always complete: the object must really be resting in the basket first
        self.retrieve = Demo(side=self.side, perception=self.perception, verbose=self.verbose, scene=self.place_in.scene)
        run_retrieve(self.retrieve, self.retrieve_to, viewer, stop_after=stop_after)


def run_retrieve_trial(task: RetrieveDemo, viewer=None, stop_after: str | None = None) -> TrialResult:
    """Run both legs and combine them into one TrialResult. Success requires the
    object to end up resting outside the basket footprint, upright, and close to
    `retrieve_to`; failure in leg 1 means the object never made it into the basket
    at all, so leg 2 never runs."""
    place_result = run_trial(task.place_in, viewer, stop_after=None)
    if not place_result.success:
        place_result.failure_reason = f"place-into-basket leg: {place_result.failure_reason}"
        return place_result

    task.retrieve = Demo(side=task.side, perception=task.perception, verbose=task.verbose, scene=task.place_in.scene)
    demo = task.retrieve
    failure = None
    try:
        run_retrieve(demo, task.retrieve_to, viewer, stop_after=stop_after)
    except RuntimeError as error:
        failure = str(error)
    scene, values = demo.scene, demo.log.values
    final_pos = scene.object_position()
    target_xy = np.array(task.retrieve_to)
    placement_error = float(np.linalg.norm(final_pos[:2] - target_xy))
    still_in_basket = scene.object_inside_basket()
    tilt = float(upright_tilt_degrees(scene.object_quaternion()))
    if failure is None and stop_after is None:
        if still_in_basket:
            failure = "object footprint is still inside the basket after set-down"
        elif not scene.object_on_work_surface():
            failure = "object is not resting on the table after set-down"
        elif tilt > C.PROOF_LIFT_MAX_TILT_DEG:
            failure = f"final object tilt {tilt:.1f}deg exceeds {C.PROOF_LIFT_MAX_TILT_DEG:.0f}deg"
    return TrialResult(
        failure is None,
        f"retrieve leg: {failure}" if failure else None,
        demo.log.current_phase if failure else None,
        final_pos.tolist(),
        float(scene.data.time),  # place_in and retrieve share one continuous MjData clock
        placement_error_m=placement_error,
        inside_basket=False,
        bottle_tilt_deg=tilt,
        contact_forces=scene.finger_contact_forces(demo.side),
        pick_position=list(task.basket_position),
        basket_position=list(task.retrieve_to),
        perception_used=task.perception,
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
        route="RETRIEVE_FROM_BASKET",
        route_reason="grasp back out of the basket the object was just placed in, set down on bare table",
        source_arm=task.place_side,
        target_arm=task.side,
        min_joint_margin_deg=values.get("min_joint_margin_deg"),
        max_penetration_m=demo.executor.max_penetration_m,
        model_timestep_s=float(scene.model.opt.timestep),
    )
