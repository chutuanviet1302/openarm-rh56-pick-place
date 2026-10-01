import inspect
import unittest

import numpy as np

import simulation.object_detector as object_detector
from simulation.object_detector import ObjectDetector, ObjectTracker
from simulation.objects import Placement
from simulation.pick_place.scene import Scene

BASKET = (0.28, 0.0)
KNOWN = ("can", "tuna_can", "gelatin_box", "orange")


def alone(key: str, yaw: float, pose: str = "upright") -> Scene:
    return Scene((0.28, -0.25), BASKET, work_platform_height=0.10, pick_object=key, pick_pose=pose, pick_yaw_deg=yaw)


class ObjectDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        layout = [Placement("orange", (0.40, -0.20)), Placement("tuna_can", (0.24, -0.36), "upright", 30.0),
                  Placement("gelatin_box", (0.38, -0.36), "upright", 20.0)]
        cls.scene = Scene((0.24, -0.20), BASKET, work_platform_height=0.10, pick_object="can", extra_objects=layout)
        cls.detector = ObjectDetector(cls.scene.model, KNOWN, basket_xy=BASKET)
        cls.detector.enroll(alone)

    def test_every_object_is_found_and_identified(self):
        detections = self.detector.detect(self.scene.data)
        self.assertEqual(sorted(d.label for d in detections), sorted(KNOWN))
        for detection in detections:
            # The visible points' centroid sits within a few cm of the object's centre.
            truth = self.scene.object_pose(detection.label)
            from simulation.objects import geometric_center
            centre = truth[:3, :3] @ geometric_center(detection.label) + truth[:3, 3]
            self.assertLess(np.linalg.norm(detection.centroid[:2] - centre[:2]), 0.03, detection.label)

    def test_detector_reads_no_simulator_object_state(self):
        source = inspect.getsource(object_detector)
        for forbidden in ("qpos", "xpos", "object_pose", "geom_xpos", "segmentation"):
            self.assertNotIn(forbidden, source)


class ObjectTrackerTest(unittest.TestCase):
    def test_ids_persist_and_vanished_objects_drop(self):
        from simulation.object_detector import Detection, Features

        def detection(label, x):
            return Detection(label, 0.0, np.zeros((2, 2), bool), np.array([x, -0.2, 0.1]),
                             Features(0, 0, 0, 0, 0, 0), 100)

        tracker = ObjectTracker(gate=0.08, max_missed=1)
        first = tracker.update([detection("can", 0.30), detection("orange", 0.40)])
        ids = {t.label: t.track_id for t in first}
        second = tracker.update([detection("can", 0.32)])  # can moved 2 cm, orange gone
        self.assertEqual([t.track_id for t in second], [ids["can"]])
        tracker.update([detection("can", 0.32)])
        self.assertNotIn(ids["orange"], tracker.tracks)  # dropped after max_missed looks


if __name__ == "__main__":
    unittest.main()
