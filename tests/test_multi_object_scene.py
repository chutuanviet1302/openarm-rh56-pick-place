import unittest

import mujoco
import numpy as np

from simulation.five_finger_model import build_five_finger_model
from simulation.objects import OBJECTS, Placement, collision_points, quat_to_matrix, spawn_height, spawn_origin_xy, spawn_quat
from simulation.pick_place.scene import Scene

PLATFORM = 0.10
BASKET = (0.28, 0.0)
PICK = (0.28, -0.25)
EXTRAS = [Placement("apple", (0.30, -0.42)), Placement("pear", (0.44, -0.22), "lying", 60.0)]


class ObjectRegistryTest(unittest.TestCase):
    def test_every_rest_pose_touches_the_surface_exactly(self):
        for key, spec in OBJECTS.items():
            for pose in spec.poses:
                placement = Placement(key, (0.0, 0.0), pose, 37.0)
                points = collision_points(key) @ quat_to_matrix(spawn_quat(placement)).T
                self.assertAlmostEqual(float(points[:, 2].min()) + spawn_height(placement), 0.0, places=9)

    def test_lying_can_rests_on_its_radius(self):
        self.assertAlmostEqual(spawn_height(Placement("can", (0, 0), "lying", 0.0)), OBJECTS["can"].cylinder[0], places=4)
        self.assertAlmostEqual(spawn_height(Placement("can", (0, 0), "upright")), OBJECTS["can"].cylinder[1], places=4)

    def test_unknown_object_or_pose_is_rejected(self):
        with self.assertRaises(ValueError):
            Placement("banana", (0.0, 0.0))
        with self.assertRaises(ValueError):
            Placement("pear", (0.0, 0.0), "upright")


class MultiObjectSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = Scene(
            pick_position=PICK, basket_position=BASKET, work_platform_height=PLATFORM,
            pick_object="can", pick_pose="lying", pick_yaw_deg=30.0, extra_objects=EXTRAS,
        )

    def test_objects_are_indexed_by_key(self):
        self.assertEqual(set(self.scene.object_joints), {"can", "apple", "pear"})
        self.assertEqual(self.scene.object_joints["can"], "pick_bottle_joint")

    def test_ground_truth_pose_matches_the_spawn(self):
        pose = self.scene.object_pose("pear")
        placement = EXTRAS[1]
        np.testing.assert_allclose(pose[:2, 3], spawn_origin_xy(placement), atol=1e-9)
        self.assertAlmostEqual(pose[2, 3], PLATFORM + spawn_height(placement), places=9)
        np.testing.assert_allclose(pose[:3, :3], quat_to_matrix(spawn_quat(placement)), atol=1e-9)
        # The lying can's axis (body z) is horizontal.
        self.assertAlmostEqual(float(self.scene.object_pose()[2, 2]), 0.0, places=6)

    def test_objects_stay_put_for_half_a_second(self):
        scene = self.scene
        scene.reset()
        start = {key: scene.object_pose(key) for key in scene.object_joints}
        for _ in range(int(0.5 / scene.model.opt.timestep)):
            mujoco.mj_step(scene.model, scene.data)
        for key, before in start.items():
            after = scene.object_pose(key)
            self.assertLess(np.linalg.norm(after[:3, 3] - before[:3, 3]), 0.002, key)
            self.assertGreater(np.trace(before[:3, :3].T @ after[:3, :3]), 1.0 + 2.0 * np.cos(np.radians(2.0)), key)
        scene.reset()

    def test_reset_restores_every_object(self):
        scene = self.scene
        spawn = {key: scene.object_pose(key) for key in scene.object_joints}
        joint = scene.model.joint("obj_apple_joint")
        scene.data.qpos[int(joint.qposadr[0])] += 0.05
        scene.reset()
        for key, pose in spawn.items():
            np.testing.assert_allclose(scene.object_pose(key), pose, atol=1e-12)


class LayoutValidationTest(unittest.TestCase):
    def test_crowded_objects_are_rejected(self):
        with self.assertRaises(ValueError):
            build_five_finger_model(pick_bottle=True, pick_position=PICK, basket_position=BASKET,
                                    work_platform_height=PLATFORM, extra_objects=[Placement("apple", (0.33, -0.28))])

    def test_object_in_the_basket_is_rejected(self):
        with self.assertRaises(ValueError):
            build_five_finger_model(pick_bottle=True, pick_position=PICK, basket_position=BASKET,
                                    work_platform_height=PLATFORM, extra_objects=[Placement("apple", (0.30, 0.02))])


if __name__ == "__main__":
    unittest.main()
