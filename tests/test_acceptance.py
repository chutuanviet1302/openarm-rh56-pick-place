from __future__ import annotations

import unittest
import numpy as np

from simulation.pick_place_demo import Demo, run_trial, sample_layout, upright_tilt_degrees


class AcceptanceTests(unittest.TestCase):
    """End-to-end acceptance tests across pick-and-place trials."""

    def test_single_trial_metrics(self):
        """Phase 5.1: Verify single pick-and-place trial metrics."""
        demo = Demo()
        demo.run()

        final_pos = demo.data.qpos[demo.bottle_qpos : demo.bottle_qpos + 3]
        basket_pos = demo.data.geom_xpos[demo.model.geom("place_basket_bottom").id]
        placement_error = float(np.linalg.norm(final_pos[:2] - basket_pos[:2]))

        tilt = upright_tilt_degrees(demo.data.qpos[demo.bottle_qpos + 3 : demo.bottle_qpos + 7])

        # Placement error must be <= 20mm
        self.assertLessEqual(placement_error, 0.020, f"Placement error {placement_error*1000:.1f}mm exceeds 20mm")
        # Final tilt must be upright < 15 degrees
        self.assertLessEqual(tilt, 15.0, f"Final tilt {tilt:.1f} deg exceeds 15 deg")

    def test_twenty_trials_acceptance(self):
        """At least 19/20 randomized RGB-D trials meet the physical metrics."""
        passed = 0
        total = 20
        errors = []

        rng = np.random.default_rng(7)
        for trial in range(total):
            demo = Demo(**sample_layout(rng, perception=True), perception=True)
            try:
                result = run_trial(demo)
                if result.success:
                    passed += 1
                else:
                    errors.append(f"Trial {trial}: {result.failure_reason}")
            except Exception as e:
                errors.append(f"Trial {trial}: {e}")

        self.assertGreaterEqual(
            passed, 19, f"Pass rate {passed}/{total} below 19/20 threshold. Failures: {errors}"
        )


if __name__ == "__main__":
    unittest.main()
