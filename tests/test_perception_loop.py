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
        self.assertEqual(overlay.current(t), 0)
        scn = mujoco.MjvScene(self.task.scene.model, maxgeom=2000)
        overlay.markers(scn, t)
        self.assertGreater(scn.ngeom, 0)
        self.assertEqual(overlay.image(t)[0], "det")


if __name__ == "__main__":
    unittest.main()
