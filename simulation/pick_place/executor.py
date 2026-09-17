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
        # Optional hook called after every physics step (used by --trace).
        self.on_step = on_step

    # ------------------------------------------------------------------ stepping
    def _step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._check_collisions()
        if self.on_step is not None:
            self.on_step(self)
        self._render()

    def _check_collisions(self) -> None:
        offenders = self.scene.table_contacts()
        if offenders:
            detail = ", ".join(f"{body} {depth:.1f}mm" for body, depth in sorted(offenders.items()))
            raise RuntimeError(f"trajectory aborted: {detail} inside the table")
        offenders = self.scene.basket_contacts()
        if offenders:
            detail = ", ".join(f"{body} {depth:.1f}mm" for body, depth in sorted(offenders.items()))
            raise RuntimeError(f"trajectory aborted: {detail} colliding with the basket")

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
        """Blend each group (`right_arm`, `right_hand`, ...) through its waypoints.

        One ease-in/ease-out envelope spans the whole multi-waypoint path with linear
        blending in between, so the motion never stops at intermediate poses.
        """
        groups = list(waypoints)
        paths = {group: [self.data.qpos[self.scene.qpos_for(group)].copy(), *waypoints[group]] for group in groups}
        cumulative = np.cumsum([0.0, *durations])
        fraction_at = cumulative / cumulative[-1]
        steps = self.seconds_to_steps(cumulative[-1])
        for index in range(steps):
            eased = quintic((index + 1) / steps)
            segment = int(np.clip(np.searchsorted(fraction_at, eased, side="right") - 1, 0, len(durations) - 1))
            span = fraction_at[segment + 1] - fraction_at[segment]
            local_t = 0.0 if span <= 0 else (eased - fraction_at[segment]) / span
            for group in groups:
                start, end = paths[group][segment], paths[group][segment + 1]
                self.data.ctrl[self.scene.ctrl_for(group)] = start + local_t * (end - start)
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

    # ------------------------------------------------------------------ proof lift
    def proof_lift(self) -> tuple[float, float, float]:
        """Lift PROOF_LIFT_HEIGHT under physics. Returns (object rise, object tilt deg,
        hand rise). Nothing is welded: if the grip is not real the object stays put."""
        scene = self.scene
        object_before = float(scene.object_position()[2])
        hand_before = float(scene.wrist_position("right")[2])
        start = self.data.ctrl[scene.arm_actuators["right"]].copy()
        target = solve_pose_ik(
            self.model, "right", scene.wrist_position("right") + np.array([0.0, 0.0, C.PROOF_LIFT_HEIGHT]),
            scene.grasp_orientation, start,
        )
        steps = self.seconds_to_steps(C.PROOF_LIFT_SECONDS)
        for index in range(steps):
            self.data.ctrl[scene.arm_actuators["right"]] = start + (target - start) * ((index + 1) / steps)
            self._step()
        rise = float(scene.object_position()[2]) - object_before
        hand_rise = float(scene.wrist_position("right")[2]) - hand_before
        return rise, upright_tilt_degrees(scene.object_quaternion()), hand_rise
