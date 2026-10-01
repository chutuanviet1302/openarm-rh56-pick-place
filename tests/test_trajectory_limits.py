import unittest

import numpy as np

from simulation.pick_place import config as C
from simulation.pick_place.executor import Executor
from simulation.pick_place.kinematics import path_spline
from simulation.pick_place.scene import Scene


class PathSplineTest(unittest.TestCase):
    def test_passes_through_every_point_in_order(self):
        points = [np.array([0.0, 0.0]), np.array([1.0, 0.5]), np.array([2.0, 0.0]), np.array([3.0, 1.0])]
        curve, travel = path_spline(points)
        chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
        u = np.concatenate([[0.0], np.cumsum(chords)]) / chords.sum()
        for s, p in zip(u, points):
            np.testing.assert_allclose(curve(s), p, atol=1e-12)
        self.assertGreaterEqual(travel, 3.0 - 1e-9)  # joint 0 goes 0 -> 3

    def test_no_joint_leaves_the_range_of_its_segment_ends(self):
        rng = np.random.default_rng(3)
        points = [rng.uniform(-1.0, 1.0, 7) for _ in range(5)]
        curve, _ = path_spline(points)
        chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
        u = np.concatenate([[0.0], np.cumsum(chords)]) / chords.sum()
        for i in range(len(points) - 1):
            low, high = np.minimum(points[i], points[i + 1]), np.maximum(points[i], points[i + 1])
            for s in np.linspace(u[i], u[i + 1], 50):
                q = curve(s)
                self.assertTrue(np.all(q >= low - 1e-12) and np.all(q <= high + 1e-12))

    def test_two_points_is_a_straight_line_and_repeats_are_dropped(self):
        a, b = np.array([0.0, 1.0]), np.array([1.0, -1.0])
        curve, travel = path_spline([a, a, b])
        np.testing.assert_allclose(curve(0.25), a + 0.25 * (b - a))
        self.assertAlmostEqual(travel, 2.0)


class FollowPathTest(unittest.TestCase):
    def test_smooth_motion_respects_the_speed_cap_and_ends_at_rest(self):
        scene = Scene()
        executor = Executor(scene)
        side = "right"
        start = scene.data.ctrl[scene.arm_actuators[side]].copy()
        offsets = [np.array([0.3, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]), np.array([0.6, 0.1, 0.0, 0.4, 0.0, 0.0, 0.0])]
        way = [start + o for o in offsets]
        commands = []
        executor.on_step = lambda ex: commands.append(ex.data.ctrl[scene.arm_actuators[side]].copy())
        seconds = executor.follow_path({f"{side}_arm": way}, 0.1)  # far too short: stretched
        commands = np.asarray(commands)
        dt = scene.model.opt.timestep
        np.testing.assert_allclose(commands[-1], way[-1], atol=1e-9)
        # Average joint speed over the move at most the cap (the duration was stretched).
        travel = np.max(np.sum(np.abs(np.diff(np.vstack([start, commands]), axis=0)), axis=0))
        self.assertLessEqual(travel / seconds, C.MAX_JOINT_SPEED_RAD_S * 1.001)
        # Starts and ends at rest: the first and last command steps are tiny.
        speed = np.abs(np.diff(np.vstack([start, commands]), axis=0)).max(axis=1) / dt
        self.assertLess(speed[0], 0.01 * speed.max())
        self.assertLess(speed[-1], 0.01 * speed.max())
        # No stop in the middle (one motion through the way point).
        middle = speed[len(speed) // 4: 3 * len(speed) // 4]
        self.assertGreater(middle.min(), 0.2 * speed.max())


if __name__ == "__main__":
    unittest.main()
