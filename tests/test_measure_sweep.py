import unittest

from scripts.measure_sweep import measure_sweep


class MeasureSweepTest(unittest.TestCase):
    def test_scans_requested_tilts_and_closure_steps(self):
        report = measure_sweep()

        self.assertEqual([row["tilt_degrees"] for row in report["summary"]], list(range(0, 91, 15)))
        self.assertEqual(len(report["samples"]), 7 * 21)
        self.assertTrue(all(len(sample["tips_wrist_m"]) == 5 for sample in report["samples"]))
        self.assertTrue(all(row["min_aperture_m"] <= row["max_aperture_m"] for row in report["summary"]))


if __name__ == "__main__":
    unittest.main()
