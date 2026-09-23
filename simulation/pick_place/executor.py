"""Execution: drive the arm and hand through `data.ctrl` under physics.

The executor is the only module that steps time. It never writes qpos/qvel of the
robot or the object -- the position servos and MuJoCo's contact solver do all the
work -- and every step is checked for table/basket collisions, which abort the
motion with the offending body named.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from simulation.pick_place import config as C
from simulation.pick_place.kinematics import quintic, solve_pose_ik, upright_tilt_degrees
from simulation.pick_place.scene import Scene


class Executor:
    def __init__(self, scene: Scene, viewer=None, on_step=None) -> None:
        self.scene = scene
        self.model, self.data = scene.model, scene.data
        self.viewer = viewer
        self._wall_anchor: float | None = None
        self._last_frame_time = -1.0
        self.max_penetration_m = 0.0
        # Optional hook called after every physics step (used by --trace).
        self.on_step = on_step
        # The arm doing the task. When set, the *other* arm touching the object is
        # a collision too (its resting pose must stay out of the working arm's way).
        self.active_side: str | None = None

    # ------------------------------------------------------------------ stepping
    def _step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self._check_collisions()
        if self.on_step is not None:
            self.on_step(self)
        self._render()

    def _check_collisions(self) -> None:
        offenders = self.scene.support_contacts()
        if offenders:
            self.max_penetration_m = max(self.max_penetration_m, max(-depth for depth in offenders.values()) / 1000.0)
            detail = ", ".join(f"{body} {depth:.1f}mm" for body, depth in sorted(offenders.items()))
            raise RuntimeError(f"trajectory aborted: {detail} inside the floor")
        offenders = self.scene.basket_contacts()
        if offenders:
            self.max_penetration_m = max(self.max_penetration_m, max(-depth for depth in offenders.values()) / 1000.0)
            detail = ", ".join(f"{body} {depth:.1f}mm" for body, depth in sorted(offenders.items()))
            raise RuntimeError(f"trajectory aborted: {detail} colliding with the basket")
        offenders = self.scene.robot_body_contacts()
        if offenders:
            detail = ", ".join(f"{body} {depth:.1f}mm" for body, depth in sorted(offenders.items()))
            raise RuntimeError(f"trajectory aborted: {detail} pressing into the robot's pedestal/torso")
        offenders = self.scene.inter_arm_contacts()
        if offenders:
            detail = ", ".join(f"{pair} {depth:.1f}mm" for pair, depth in sorted(offenders.items()))
            raise RuntimeError(f"trajectory aborted: the two arms touch ({detail})")
        if self.active_side is not None:
            idle = "left" if self.active_side == "right" else "right"
            for contact in self.data.contact[: self.data.ncon]:
                geoms = (contact.geom1, contact.geom2)
                if self.scene.object_geom in geoms and float(contact.dist) < 0.0:
                    other = geoms[1] if geoms[0] == self.scene.object_geom else geoms[0]
                    if self.scene.robot_side(other) == idle:
                        body = self.model.body(int(self.model.geom_bodyid[other])).name
                        raise RuntimeError(f"trajectory aborted: idle {idle} arm ({body}) touches the object")

    def _render(self) -> None:
        """Redraw once per frame and pace to real time (sync + sleep after *every* 1ms
        step ran at ~0.4x: viewer.sync() costs ~1ms and Windows cannot sleep < ~1.6ms)."""
        if self.viewer is None:
            return
        now = time.perf_counter()
        sim_time = float(self.data.time)
        if self._wall_anchor is None:
            self._wall_anchor = now - sim_time
        if sim_time - self._last_frame_time >= C.VIEWER_FRAME_SECONDS:
            self.viewer.sync()
            self._last_frame_time = sim_time
            now = time.perf_counter()
            ahead = self._wall_anchor + sim_time - now
            if ahead > 0.0:
                time.sleep(ahead)
            elif ahead < -0.5:
                self._wall_anchor = now - sim_time  # fell far behind: re-anchor, don't race

    def seconds_to_steps(self, seconds: float) -> int:
        return max(1, int(seconds / self.model.opt.timestep))

    def hold(self, seconds: float) -> None:
        """Step physics while holding the current ctrl targets."""
        for _ in range(self.seconds_to_steps(seconds)):
            self._step()

    # ------------------------------------------------------------------ motion
    def move_to(self, targets: dict[str, np.ndarray], seconds: float) -> None:
        self.follow({group: [target] for group, target in targets.items()}, [seconds])

    def follow(self, waypoints: dict[str, list[np.ndarray]], durations: list[float]) -> None:
        """Follow waypoints with continuous joint velocity and zero-speed endpoints."""
        groups = list(waypoints)
        paths = {group: [self.data.qpos[self.scene.qpos_for(group)].copy(), *waypoints[group]] for group in groups}
        # Joint speed limit: stretch any segment whose largest joint move would exceed
        # MAX_JOINT_SPEED_RAD_S on average. Without it a 0.18rad waypoint step in a
        # 0.09s slot flung a held can out of the hand (7cm wrist jump in 0.15s at the
        # start of a carry, 2026-09-23) -- and it caps the DM motors' commanded speed.
        # Arm joints only: the hand's own open/close timing is part of the tuned
        # release (fingers uncurl while the wrist retreats; slowing them to the arm's
        # cap shoved the can 14mm on release).
        arms = [group for group in groups if group.endswith("_arm")]
        durations = [
            max(float(duration), max(
                (float(np.max(np.abs(paths[group][index + 1] - paths[group][index]))) for group in arms), default=0.0
            ) / C.MAX_JOINT_SPEED_RAD_S)
            for index, duration in enumerate(durations)
        ]
        cumulative = np.cumsum([0.0, *durations])
        steps = self.seconds_to_steps(cumulative[-1])
        for index in range(steps):
            elapsed = (index + 1) * cumulative[-1] / steps
            segment = int(np.clip(np.searchsorted(cumulative, elapsed, side="right") - 1, 0, len(durations) - 1))
            local_t = (elapsed - cumulative[segment]) / durations[segment]
            for group in groups:
                points = paths[group]
                start, end = points[segment], points[segment + 1]
                if len(durations) == 1:
                    target = start + quintic(local_t) * (end - start)
                else:
                    before = np.zeros_like(start) if segment == 0 else (
                        points[segment + 1] - points[segment - 1]
                    ) / (cumulative[segment + 1] - cumulative[segment - 1])
                    after = np.zeros_like(end) if segment + 1 == len(points) - 1 else (
                        points[segment + 2] - points[segment]
                    ) / (cumulative[segment + 2] - cumulative[segment])
                    u, dt = local_t, durations[segment]
                    target = (
                        (2*u**3 - 3*u**2 + 1) * start
                        + (u**3 - 2*u**2 + u) * dt * before
                        + (-2*u**3 + 3*u**2) * end
                        + (u**3 - u**2) * dt * after
                    )
                self.data.ctrl[self.scene.ctrl_for(group)] = target
            self._step()

    # ------------------------------------------------------------------ hand
    def preshape_hand(self, side: str) -> None:
        """Open the four fingers and swing the thumb into opposition *before* the arm
        travels (rh56_controller's thumb-reflex ordering), so the hand arrives straddling
        the object."""
        scene = self.scene
        for name, actuator in scene.finger_actuator[side].items():
            self.data.ctrl[actuator] = scene.open_ctrl[side][name]
        yaw_actuator, _, opposed = scene.thumb_yaw[side]
        self.data.ctrl[yaw_actuator] = opposed

    def open_fingers(self, side: str, fingers: tuple[str, ...], seconds: float, release_thumb_yaw: bool = False) -> None:
        """Ramp the named fingers' ctrl to their open values over `seconds`; with
        `release_thumb_yaw` the thumb also swings out of opposition."""
        scene = self.scene
        targets = {scene.finger_actuator[side][name]: scene.open_ctrl[side][name] for name in fingers}
        if release_thumb_yaw:
            yaw_actuator, unopposed, _ = scene.thumb_yaw[side]
            targets[yaw_actuator] = unopposed
        start = {actuator: float(self.data.ctrl[actuator]) for actuator in targets}
        steps = self.seconds_to_steps(seconds)
        for index in range(steps):
            fraction = (index + 1) / steps
            for actuator, target in targets.items():
                self.data.ctrl[actuator] = start[actuator] + (target - start[actuator]) * fraction
            self._step()

    def close_until_contact(self, side: str, fingers: tuple[str, ...], force_target: float = C.CONTACT_FORCE_TARGET_N) -> dict[str, float]:
        """Close `fingers` a small step at a time, holding each once it presses with
        `force_target`. Fingers are driven only through ctrl; the contact solver is what
        stops them on the object's surface."""
        scene = self.scene
        actuators = {name: scene.finger_actuator[side][name] for name in fingers}
        held: set[str] = set()
        settle = self.seconds_to_steps(C.CLOSE_SETTLE_SECONDS)
        for _ in range(C.CLOSE_MAX_ITERATIONS):
            forces = scene.finger_contact_forces(side)
            for name, actuator in actuators.items():
                if name in held:
                    continue
                if forces[name] >= force_target:
                    held.add(name)
                    continue
                lower, upper = self.model.actuator_ctrlrange[actuator]
                step = C.CLOSE_STEP_FRACTION * (upper - lower)
                current = float(self.data.ctrl[actuator])
                delta = float(np.clip(scene.closed_ctrl[side][name] - current, -step, step))
                self.data.ctrl[actuator] = np.clip(current + delta, lower, upper)
            if len(held) == len(actuators):
                break
            for _ in range(settle):
                self._step()
        for _ in range(settle * 4):  # settle at the final pressure
            self._step()
        return scene.finger_contact_forces(side)

    # ------------------------------------------------------------------ set-down
    def descend_until(self, side: str, orientation: np.ndarray, stop, max_depth: float = C.SET_DOWN_MAX_DEPTH) -> float:
        """Lower the wrist straight down in SET_DOWN_STEP increments until `stop()` is
        true (e.g. the object touches the basket floor) or `max_depth` is reached.
        Returns how far the wrist went down. Collisions of the hand with the basket
        still abort through _step()."""
        scene = self.scene
        seed = self.data.ctrl[scene.arm_actuators[side]].copy()
        start_z = float(scene.wrist_position(side)[2])
        target = scene.wrist_position(side).copy()
        descended = 0.0
        while not stop() and descended < max_depth:
            target[2] = start_z - descended - C.SET_DOWN_STEP
            try:
                seed = solve_pose_ik(self.model, side, target, orientation, seed)
            except RuntimeError:
                break  # at the arm's reach; the caller judges whether the object is down
            descended += C.SET_DOWN_STEP
            self.move_to({f"{side}_arm": seed}, C.SET_DOWN_STEP_SECONDS)
        return descended

    # ------------------------------------------------------------------ proof lift
    def proof_lift(self, side: str, orientation: np.ndarray) -> tuple[float, float, float]:
        """Lift PROOF_LIFT_HEIGHT under physics. Returns (object rise, object tilt deg,
        hand rise). Nothing is welded: if the grip is not real the object stays put."""
        scene = self.scene
        object_before = float(scene.object_position()[2])
        hand_before = float(scene.wrist_position(side)[2])
        start = self.data.ctrl[scene.arm_actuators[side]].copy()
        target = solve_pose_ik(
            self.model, side, scene.wrist_position(side) + np.array([0.0, 0.0, C.PROOF_LIFT_HEIGHT]),
            orientation, start,
        )
        steps = self.seconds_to_steps(C.PROOF_LIFT_SECONDS)
        for index in range(steps):
            self.data.ctrl[scene.arm_actuators[side]] = start + (target - start) * ((index + 1) / steps)
            self._step()
        rise = float(scene.object_position()[2]) - object_before
        hand_rise = float(scene.wrist_position(side)[2]) - hand_before
        return rise, upright_tilt_degrees(scene.object_quaternion()), hand_rise
