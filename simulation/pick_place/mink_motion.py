"""Optimization-based arm motion with mink (differential IK as a QP).

Instead of solving each waypoint's IK separately and blending joints between them,
every control tick solves one small quadratic program for the arm's joint velocity:

    minimise   |J v - wrist error|^2 (FrameTask)  +  w |q - q_posture|^2 (PostureTask)
    subject to joint position limits (ConfigurationLimit) and -- optionally -- a
               minimum distance between the
               hand/forearm and the table / box (CollisionAvoidanceLimit)

so limits and clearance are constraints of the motion itself, not checks run after a
plan. The result is written to the arm's position actuators through the Executor, so
the servo speed cap, contact checks and recording stay exactly as for scripted moves.

    arm = MinkArm(scene, "right", avoid=scene.table_geoms | scene.basket_geoms)
    arm.move_to_pose(executor, target_4x4, seconds=1.5)       # Cartesian straight line
    q = arm.step(target_4x4)                                  # one tick (policy control)
"""

from __future__ import annotations

import mink
import mujoco
import numpy as np

from simulation.pick_place import config as C

SOLVER = "daqp"
TICK_S = 0.01
POSITION_COST = 1.0
ORIENTATION_COST = C.IK_ROTATION_WEIGHT
POSTURE_COST = 1e-2
DAMPING = 1e-3
AVOID_DISTANCE_M = 0.01


def _pose(target: np.ndarray) -> mink.SE3:
    return mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(np.asarray(target[:3, :3])), np.asarray(target[:3, 3]))


class MinkArm:
    def __init__(self, scene, side: str, avoid: set[int] | None = None) -> None:
        self.scene, self.side = scene, side
        model = scene.model
        self.configuration = mink.Configuration(model)
        self.frame = mink.FrameTask(C.EE_SITE[side], "site", position_cost=POSITION_COST,
                                    orientation_cost=ORIENTATION_COST, lm_damping=1.0)
        self.posture = mink.PostureTask(model, cost=POSTURE_COST)
        # Only this arm moves. Not a DofFreezingTask: a frozen hand joint resting a hair
        # outside its range made the QP infeasible (the limit asks it to move). The other
        # joints do not move the wrist, so the QP leaves them be; their velocity is
        # dropped before integrating.
        self.mask = np.zeros(model.nv)
        self.mask[np.asarray(scene.arm_dofs[side], dtype=int)] = 1.0
        self.limits: list = [mink.ConfigurationLimit(model)]
        if avoid:
            own = [g for g in range(model.ngeom) if scene.robot_side(g) == side
                   and (model.geom_contype[g] or model.geom_conaffinity[g])]
            self.limits.append(mink.CollisionAvoidanceLimit(
                model, [(own, sorted(avoid))], minimum_distance_from_collisions=AVOID_DISTANCE_M,
                collision_detection_distance=AVOID_DISTANCE_M + 0.03))
        self._posture_set = False

    def _sync(self, arm_q: np.ndarray | None = None) -> None:
        """The QP starts from the world as it is, with the arm at its current command
        (the servos lag the command by a few mm; planning from the command keeps the
        commanded trajectory continuous, as Executor.follow does)."""
        scene, side = self.scene, self.side
        q = scene.data.qpos.copy()
        q[scene.arm_qpos[side]] = scene.data.ctrl[scene.arm_actuators[side]] if arm_q is None else arm_q
        self.configuration.update(q)
        if not self._posture_set:
            self.posture.set_target(q)
            self._posture_set = True

    def step(self, target: np.ndarray, dt: float = TICK_S, arm_q: np.ndarray | None = None) -> np.ndarray:
        """One QP tick toward the wrist pose `target` (4x4 world); the arm joints after it."""
        self._sync(arm_q)
        self.frame.set_target(_pose(target))
        velocity = mink.solve_ik(self.configuration, [self.frame, self.posture], dt, SOLVER, damping=DAMPING,
                                 limits=self.limits)
        self.configuration.integrate_inplace(velocity * self.mask, dt)
        return self.configuration.q[self.scene.arm_qpos[self.side]].copy()

    def converge(self, target: np.ndarray, arm_q: np.ndarray | None = None, iterations: int = 200,
                 tolerance_m: float = C.IK_POSITION_TOLERANCE) -> np.ndarray:
        """Iterate the QP on the model only (no physics) until the wrist reaches `target`."""
        q = self.scene.data.ctrl[self.scene.arm_actuators[self.side]].copy() if arm_q is None else arm_q.copy()
        for _ in range(iterations):
            q = self.step(target, dt=0.05, arm_q=q)
            error = self.frame.compute_error(self.configuration)
            if np.linalg.norm(error[:3]) < tolerance_m and np.linalg.norm(error[3:]) < C.IK_ROTATION_TOLERANCE:
                break
        return q

    def move_to_pose(self, executor, target: np.ndarray, seconds: float, segments: int = 10) -> None:
        """Cartesian straight line (position lerp, rotation slerp) from the commanded wrist
        pose to `target`, each segment end solved by the QP from the previous one; the
        executor follows the joint path (speed cap, contact checks)."""
        from simulation.pick_place.kinematics import wrist_frame

        scene, side = self.scene, self.side
        q = scene.data.ctrl[scene.arm_actuators[side]].copy()
        start_p, start_r = wrist_frame(scene.model, side, q)
        start, goal = mink.SO3.from_matrix(start_r), mink.SO3.from_matrix(np.asarray(target[:3, :3]))
        delta = (start.inverse() @ goal).log()
        path = []
        for k in range(1, segments + 1):
            f = k / segments
            waypoint = np.eye(4)
            waypoint[:3, :3] = (start @ mink.SO3.exp(delta * f)).as_matrix()
            waypoint[:3, 3] = start_p + f * (np.asarray(target[:3, 3]) - start_p)
            q = self.converge(waypoint, arm_q=q, iterations=60)
            path.append(q)
        executor.follow({f"{side}_arm": path}, [seconds / segments] * segments)
