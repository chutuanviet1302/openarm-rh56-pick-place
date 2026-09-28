import unittest

import numpy as np

from simulation.objects import Placement, quat_to_matrix, spawn_quat
from simulation.pick_place.grasp_library import jaw_heading_deg, load_library, select_grasp
from simulation.pick_place.pose_source import pose_error


def pose_of(key: str, pose: str = "upright", yaw: float = 0.0, xyz=(0.3, -0.2, 0.1)) -> np.ndarray:
    placement = Placement(key, xyz[:2], pose, yaw)
    transform = np.eye(4)
    transform[:3, :3] = quat_to_matrix(spawn_quat(placement))
    transform[:3, 3] = xyz
    return transform


class GraspLibraryTest(unittest.TestCase):
    def test_every_registry_pose_in_the_library_has_a_grasp(self):
        for key, pose in (("can", "upright"), ("can", "lying"), ("apple", "upright"),
                          ("orange", "upright"), ("pear", "lying")):
            for yaw in (0.0, 37.0, 150.0):
                target = select_grasp(key, pose_of(key, pose, yaw))
                self.assertEqual(target.object_key, key)
                self.assertAlmostEqual(target.tilt_deg(pose_of(key, pose, yaw)[:3, :3]), 0.0, places=6)

    def test_rest_pose_picks_the_grasp(self):
        self.assertEqual(select_grasp("can", pose_of("can", "upright")).name, "upright_wrap")
        self.assertEqual(select_grasp("can", pose_of("can", "lying", 20.0)).name, "lying_wrap")

    def test_lying_can_jaw_runs_across_its_axis(self):
        for yaw in (0.0, 30.0, 90.0, 135.0):
            target = select_grasp("can", pose_of("can", "lying", yaw))
            axis = target.pose[:3, 2]
            jaw = np.radians(target.jaw_heading_deg)
            self.assertAlmostEqual(float(np.dot(axis[:2], [np.cos(jaw), np.sin(jaw)])), 0.0, places=6)

    def test_round_objects_take_any_heading(self):
        self.assertIsNone(select_grasp("can", pose_of("can", "upright", 50.0)).jaw_heading_deg)
        self.assertIsNone(select_grasp("apple", pose_of("apple", "upright", 50.0)).jaw_heading_deg)
        self.assertIsNone(jaw_heading_deg(np.eye(3), (0.0, 0.0, 1.0)))

    def test_grasp_centre_follows_the_pose(self):
        pose = pose_of("pear", "lying", 90.0)
        target = select_grasp("pear", pose)
        np.testing.assert_allclose(target.center_world, pose[:3, :3] @ [0.0, -0.01, 0.0] + pose[:3, 3], atol=1e-12)

    def test_pose_no_entry_covers_is_refused(self):
        standing_pear = pose_of("pear", "lying")
        standing_pear[:3, :3] = standing_pear[:3, :3] @ quat_to_matrix((np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0))
        with self.assertRaises(RuntimeError):
            select_grasp("pear", standing_pear)
        with self.assertRaises(RuntimeError):
            select_grasp("banana", np.eye(4))

    def test_lying_can_tilt_is_its_axis_off_the_table(self):
        target = select_grasp("can", pose_of("can", "lying"))
        rotation = pose_of("can", "lying")[:3, :3]
        # Its axis lies along world y: tipping about world x lifts one end.
        tipped = quat_to_matrix((np.cos(np.radians(5)), np.sin(np.radians(5)), 0.0, 0.0)) @ rotation
        rolled = rotation @ quat_to_matrix((np.cos(np.radians(40)), 0.0, 0.0, np.sin(np.radians(40))))
        self.assertAlmostEqual(target.tilt_deg(rolled), 0.0, places=6)  # rolling about its axis is not tilting
        self.assertGreater(target.tilt_deg(tipped), 1.0)

    def test_library_file_parses(self):
        library = load_library()
        self.assertEqual(set(library), {"can", "apple", "orange", "pear", "tuna_can", "tennis_ball", "gelatin_box", "peach"})


class PoseErrorTest(unittest.TestCase):
    def test_symmetry_aware_rotation_error(self):
        truth = pose_of("can", "upright")
        spun = truth.copy()
        spun[:3, :3] = truth[:3, :3] @ quat_to_matrix((np.cos(np.radians(45)), 0.0, 0.0, np.sin(np.radians(45))))
        self.assertAlmostEqual(pose_error(spun, truth, "axial")[1], 0.0, places=6)
        self.assertAlmostEqual(pose_error(spun, truth, "none")[1], 90.0, places=6)
        self.assertEqual(pose_error(spun, truth, "spherical")[1], 0.0)


if __name__ == "__main__":
    unittest.main()
