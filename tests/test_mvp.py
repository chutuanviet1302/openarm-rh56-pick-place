import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from openarm_pick_place.fakes import FakeHand, FakeRobot
from openarm_pick_place.geometry import inverse, load_calibration, make_transform, transform_pose
from openarm_pick_place.models import CameraIntrinsics, GraspConfig, ObjectPose, Pose, Workspace
from openarm_pick_place.motion import joint_trajectory, pick_place_waypoints
from openarm_pick_place.perception import estimate_object_pose
from openarm_pick_place.pipeline import PickPlaceController, append_trial
from openarm_pick_place.ros2_nodes import inspire_command_positions


class MvpTests(unittest.TestCase):
    def test_transform_round_trip(self):
        transform = make_transform(np.eye(3), np.array([0.1, -0.2, 0.3]))
        pose = Pose(np.array([0.4, 0.5, 0.6]))
        restored = transform_pose(inverse(transform), transform_pose(transform, pose))
        np.testing.assert_allclose(restored.position, pose.position)

    def test_red_object_depth_is_deprojected(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[2:8, 3:9, 0] = 255
        depth = np.zeros((10, 10), dtype=float)
        depth[2:8, 3:9] = 1.0
        pose = estimate_object_pose(rgb, depth, CameraIntrinsics(100, 100, 5, 5), [0, 200, 200], [5, 255, 255], minimum_points=20)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose.pose.position, [0.005, -0.005, 1.0])

    def test_object_orientation_follows_mask_principal_axis(self):
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb[4:16, 8:12, 0] = 255
        depth = np.ones((20, 20), dtype=float)
        pose = estimate_object_pose(rgb, depth, CameraIntrinsics(100, 100, 10, 10), [0, 200, 200], [5, 255, 255])
        direction = pose.pose.rotation[:2, 0]
        self.assertGreater(abs(direction[1]), 0.99)

    def test_pick_place_waypoints_are_ordered(self):
        grasp = GraspConfig(np.array([0, 0, 0.1]), np.array([0, 0, 0.1]))
        waypoints = pick_place_waypoints(Pose(np.array([0, 0, 0.2])), Pose(np.array([0.5, 0, 0.2])), grasp)
        self.assertEqual(tuple(waypoints), ("pregrasp", "grasp", "lift", "preplace", "place", "retreat"))
        np.testing.assert_allclose(waypoints["lift"].position, [0, 0, 0.38])

    def test_inspire_left_command_uses_rh56_driver_order(self):
        command = inspire_command_positions([1000, 800, 600, 400, 200, 0], "left")
        self.assertEqual(len(command), 12)
        np.testing.assert_allclose(command[:6], np.full(6, np.pi))
        np.testing.assert_allclose(command[6:], np.array([1, .8, .6, .4, .2, 0]) * np.pi)

    def test_trajectory_respects_velocity(self):
        trajectory = joint_trajectory([0], [1], [0.5], 0.1)
        self.assertLessEqual(np.max(np.abs(np.diff(trajectory[:, 0]))) / 0.1, 0.5 + 1e-12)
        np.testing.assert_allclose(trajectory[[0, -1], 0], [0, 1])

    def test_pick_place_success_and_safe_abort(self):
        workspace = Workspace(np.array([-1, -1, 0]), np.array([1, 1, 2]))
        config = GraspConfig(np.array([0, 0, 0.1]), np.array([0, 0, 0.1]))
        robot, hand = FakeRobot(), FakeHand()
        controller = PickPlaceController(robot, hand, workspace, config, Pose(np.array([0.5, 0.5, 0.2])))
        result = controller.run(ObjectPose(Pose(np.array([0, 0, 0.2])), time.time()))
        self.assertTrue(result.success)
        self.assertEqual(len(robot.poses), 6)
        self.assertEqual(hand.commands, ["BOTTLE_GRASP", "OPEN"])
        stale = controller.run(ObjectPose(Pose(np.array([0, 0, 0.2])), 0))
        self.assertEqual(stale.final_state, "SAFE_ABORT")
        self.assertTrue(robot.stopped)

    def test_calibration_requires_hardware_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps({"base_from_camera": np.eye(4).tolist()}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "camera_serial"):
                load_calibration(path)

    def test_trial_csv_has_failure_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            append_trial(path, self._failed_result())
            self.assertIn("invalid depth", path.read_text(encoding="utf-8"))

    @staticmethod
    def _failed_result():
        from openarm_pick_place.models import TrialResult
        return TrialResult(False, "SAFE_ABORT", "invalid depth", 0.1)


if __name__ == "__main__":
    unittest.main()
