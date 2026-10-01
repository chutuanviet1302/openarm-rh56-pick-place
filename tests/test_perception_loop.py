import unittest

import mujoco
import numpy as np

from simulation.pick_place.bin_conveyor_task import TABLE_OBJECTS, BinConveyorTask
from simulation.pick_place.perception_loop import DetectionReplayOverlay


class PerceptionLoopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = BinConveyorTask(pose_backend="gt")
        cls.task.enroll()
        cls.look = cls.task.perception.look()

    @classmethod
    def tearDownClass(cls):
        cls.task.perception.close()

    def test_every_table_object_is_detected_and_labelled(self):
        scene = self.task.scene
        table = self.look.detections["table"]
        for placement in TABLE_OBJECTS:
            here = scene.object_position_of(placement.label)[:2]
            nearest = min(table, key=lambda d: float(np.linalg.norm(d.centroid[:2] - here)))
            self.assertLess(float(np.linalg.norm(nearest.centroid[:2] - here)), 0.03, placement.label)
            self.assertEqual(nearest.label, placement.key)
            self.assertIsNotNone(nearest.track_id)

    def test_periodic_looks_follow_the_recorder_clock(self):
        task = self.task
        looks_before = len(task.perception.log.times)
        task._settle(1.2)
        # 1.2 s of sim time at one look per 0.5 s: two more looks.
        self.assertEqual(len(task.perception.log.times) - looks_before, 2)

    def test_no_robot_pixels_and_no_unidentified_objects(self):
        look = self.task.perception.look()
        self.assertGreater(int(look.frame.robot.sum()), 0)
        for detections in look.detections.values():
            for d in detections:
                self.assertIsNotNone(d.label)
                self.assertFalse(np.any(d.mask & look.frame.robot))

    def test_held_object_follows_the_hand_and_hides_its_partial_views(self):
        perception = self.task.perception
        scene = self.task.scene
        phase = {"value": "right reach"}
        perception.phase_source = lambda: phase["value"]
        try:
            look = perception.look()
            can = next(d for d in look.detections["table"] if d.label == "can")
            perception.set_target("right", can)
            phase["value"] = "right grasp"
            perception.latest = None
            first = perception.look()
            self.assertEqual(len(first.held), 1)
            held, box = first.held[0]
            np.testing.assert_allclose(box, can.box, atol=0.01)   # where the camera saw it
            rows = [r for r in perception.log.rows if r[0] == len(perception.log.times) - 1]
            self.assertEqual([r[2] for r in rows].count("can"), 1)  # the held box only, no duplicate
            self.assertEqual([r[1] for r in rows if r[2] == "can"], ["held"])
            # The hand moves (the arm's joints): the box moves with it.
            arm = scene.arm_qpos["right"]
            saved = scene.data.qpos[arm].copy()
            scene.data.qpos[arm[1]] += 0.2
            mujoco.mj_kinematics(scene.model, scene.data)
            from simulation.pick_place.perception_loop import hand_pose

            position, rotation = hand_pose(scene.model, scene.data, "right")
            moved = held.box(position, rotation)
            self.assertGreater(float(np.linalg.norm(moved.mean(axis=0) - box.mean(axis=0))), 0.01)
            scene.data.qpos[arm] = saved
            mujoco.mj_forward(scene.model, scene.data)
            phase["value"] = "right release"
            perception.latest = None
            self.assertEqual(perception.look().held, [])
        finally:
            perception.phase_source = lambda: self.task.recorder.phase
            perception.set_target("right", None)

    def test_log_round_trips_into_the_replay_overlay(self):
        arrays = self.task.perception.log.arrays()
        n = len(arrays["det_label"])
        self.assertGreaterEqual(n, len(TABLE_OBJECTS))
        self.assertEqual(arrays["det_box"].shape, (n, 2, 3))
        self.assertTrue(np.all(arrays["det_box"][:, 0] <= arrays["det_box"][:, 1]))
        self.assertEqual(len(arrays["det_images"]), len(arrays["det_look_times"]))

        class Recording(dict):
            files = property(lambda self: list(self))

        overlay = DetectionReplayOverlay(Recording(arrays))
        t = float(arrays["det_look_times"][0])
        self.assertEqual(float(overlay.times[overlay.current(t)]), t)  # the latest look at that time
        scn = mujoco.MjvScene(self.task.scene.model, maxgeom=2000)
        overlay.markers(scn, t)
        self.assertGreater(scn.ngeom, 0)
        self.assertEqual(overlay.image(t)[0], "det")


if __name__ == "__main__":
    unittest.main()
