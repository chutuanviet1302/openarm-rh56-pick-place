import unittest

from product.kpi import pick_cycles


class PickCycleTest(unittest.TestCase):
    def test_each_pick_runs_to_the_end_of_its_outcome_caption(self):
        segments = [
            {"caption": "right arm -> can (can)", "start_s": 0.0, "seconds": 20.0},
            {"caption": "right arm: can in the box", "start_s": 20.0, "seconds": 3.0},
            {"caption": "left arm -> apple (apple)", "start_s": 23.0, "seconds": 15.0},
            {"caption": "conveyor running", "start_s": 38.0, "seconds": 4.0},
            {"caption": "conveyor: right arm -> can (can_belt)", "start_s": 42.0, "seconds": 18.0},
            {"caption": "left arm -> apple (apple)", "start_s": 60.0, "seconds": 10.0},
        ]
        cycles = pick_cycles(segments, 75.0)
        self.assertEqual([(c["name"], c["seconds"]) for c in cycles],
                         [("can", 23.0), ("apple", 15.0), ("can_belt", 18.0), ("apple", 10.0)])
        self.assertEqual(cycles[2]["kind"], "can")


if __name__ == "__main__":
    unittest.main()
