from types import SimpleNamespace
import unittest

import numpy as np

from openarm_pick_place.mujoco_bridge import arm_action_topics, trajectory_to_executor
from simulation.pick_place.config import ARM_JOINTS


class MujocoBridgeTests(unittest.TestCase):
    def test_trajectory_is_reordered_and_converted_to_segment_durations(self):
        names = list(reversed(ARM_JOINTS["right"]))
        points = [
            SimpleNamespace(positions=list(range(7)), time_from_start=SimpleNamespace(sec=1, nanosec=0)),
            SimpleNamespace(positions=list(range(10, 17)), time_from_start=SimpleNamespace(sec=3, nanosec=0)),
        ]
        waypoints, durations = trajectory_to_executor(names, points, ARM_JOINTS["right"])
        np.testing.assert_array_equal(waypoints[0], list(reversed(range(7))))
        self.assertEqual(durations, [1.0, 2.0])

    def test_rejects_non_monotonic_time(self):
        point = SimpleNamespace(positions=[0] * 7, time_from_start=SimpleNamespace(sec=0, nanosec=0))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            trajectory_to_executor(list(ARM_JOINTS["right"]), [point], ARM_JOINTS["right"])

    def test_bimanual_action_topics_and_legacy_right_topic(self):
        topics = arm_action_topics({"ros": {"arm_actions": {"right": "/right", "left": "/left"}}})
        self.assertEqual(topics, {"right": "/right", "left": "/left"})
        self.assertEqual(arm_action_topics({"ros": {"arm_action": "/legacy"}}), {"right": "/legacy"})


if __name__ == "__main__":
    unittest.main()
