import unittest

from simulation.pick_place.retrieve import RetrieveDemo, run_retrieve_trial

# Real robot height (top 0.78m above the table, re-measured 2026-09-23). The right
# arm picks at its default point, places into a basket on its own side, grasps the
# can back out from inside the walls and sets it down beside the pick point. Measured
# on this layout: set-down error 27mm, tilt 0deg, no penetration of table/basket.
PICK_A = (0.08, -0.38)
BASKET = (0.25, -0.25)
RETRIEVE_TO = (0.15, -0.40)


class RetrieveFromBasketTests(unittest.TestCase):
    def test_place_then_retrieve_from_basket(self):
        """One arm places the object in the basket, then grasps it back out from
        inside the walls (not a bare-table pick relabelled) and sets it down
        elsewhere. Both legs run in the same MjModel/MjData (Demo(scene=...)), so the
        second grasp is really contending with the basket the first leg built.

        The cross-arm version (right places, left retrieves) has no robust layout on
        the real-height robot: the right arm sets down no further left than y~+0.04,
        the left arm grasps from inside a basket only from y~+0.03, and a 2mm change
        in where the can lands flips the plan (context_project.md, 2026-09-23). Its
        earlier tests relied on a rotated arm mount the real robot does not have."""
        task = RetrieveDemo(PICK_A, BASKET, RETRIEVE_TO, side="right")
        result = run_retrieve_trial(task)
        self.assertTrue(result.success, result.failure_reason)
        self.assertLess(result.placement_error_m, 0.05)
        self.assertLess(result.bottle_tilt_deg, 5.0)
        self.assertEqual(result.max_penetration_m, 0.0)
        # A successful result already implies a real antipodal pinch during the
        # grasp (demo.py's fingers_not_pressing gate, checked before proof lift).
        self.assertGreaterEqual(result.grasp_forces["thumb"], 6.0)

    def test_right_places_left_retrieves_centre_basket_on_platform(self):
        """The cross-arm task with the basket at the table centre, on the real-height
        robot: a 10cm work platform (table-long, half the table deep, x 0.17..0.67)
        carries the object, the basket (flat floor, centred on the torso axis) and the
        set-down spot. Up there the steep top grasp pins the joints, so both arms use
        the oblique grasp (fingers ~38deg below horizontal, thumb and fingers pinching
        level). Every safety abort is live: arm/hand contact with the table, platform,
        basket or torso, the two arms touching, the idle arm touching the can. Measured
        2026-09-24: 7/8 with the pick moved +-1.5cm; this nominal run 3mm, 0deg."""
        task = RetrieveDemo(
            (0.28, -0.25), (0.28, 0.0), (0.28, 0.25), side="left", place_side="right", work_platform_height=0.10,
        )
        result = run_retrieve_trial(task)
        self.assertTrue(result.success, result.failure_reason)
        self.assertEqual((result.source_arm, result.target_arm), ("right", "left"))
        self.assertLess(result.placement_error_m, 0.02)
        self.assertLess(result.bottle_tilt_deg, 5.0)
        self.assertEqual(result.max_penetration_m, 0.0)

    def test_place_leg_failure_is_not_masked_as_a_retrieve_failure(self):
        """An unreachable pick position must fail in the place-into-basket leg
        with its own reason, not be silently swallowed or blamed on retrieval."""
        task = RetrieveDemo((0.60, -0.60), BASKET, RETRIEVE_TO, side="right")
        result = run_retrieve_trial(task)
        self.assertFalse(result.success)
        self.assertIn("place-into-basket leg", result.failure_reason)


if __name__ == "__main__":
    unittest.main()
