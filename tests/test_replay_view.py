import unittest

import mujoco
import numpy as np

from simulation.pick_place.replay_view import interpolate, quaternion_slots


class InterpolateTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><joint type='hinge'/><geom size='.1'/></body>"
            "<body><freejoint/><geom size='.1'/></body></worldbody></mujoco>")
        self.quats = quaternion_slots(self.model)

    def test_quaternion_slots_of_a_free_joint(self):
        self.assertEqual(self.quats, [4])

    def test_midpoint_is_linear_for_hinges_and_slerp_for_quaternions(self):
        half = np.pi / 4  # 90 deg about z
        a = np.array([0.0, 0, 0, 0, 1, 0, 0, 0])
        b = np.array([1.0, 2, 0, 0, np.cos(half), 0, 0, np.sin(half)])
        q, index = interpolate(np.stack([a, b]), np.array([0.0, 1.0]), 0.5, self.quats)
        self.assertEqual(index, 0)
        np.testing.assert_allclose(q[:4], [0.5, 1.0, 0, 0])
        np.testing.assert_allclose(q[4:], [np.cos(half / 2), 0, 0, np.sin(half / 2)], atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(q[4:])), 1.0)

    def test_outside_the_recording_clamps(self):
        frames = np.stack([np.zeros(8), np.ones(8)])
        frames[:, 4] = 1.0
        frames[:, 5:] = 0.0
        q, index = interpolate(frames, np.array([0.0, 1.0]), 5.0, self.quats)
        self.assertEqual(index, 1)
        np.testing.assert_allclose(q, frames[1])


if __name__ == "__main__":
    unittest.main()
