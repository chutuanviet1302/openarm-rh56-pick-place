import unittest

from simulation.pick_place.retrieve import RetrieveDemo, run_retrieve_trial

# A staging point (A) and basket ("giữa bàn") position (M) both inside the left
# arm's narrow fresh-grasp region (mapped empirically: roughly x in [0.02, 0.18],
# y in [0.28, 0.40] -- outside it plan_pick finds no reachable heading at all,
# independent of the basket). C is a set-down point with enough room around it for
# the return-to-attention path; a point too close to the pedestal column left no
# collision-free raise waypoint even though the set-down itself had succeeded.
PICK_A = (0.02, 0.28)
BASKET_MIDDLE = (0.16, 0.38)
RETRIEVE_TO = (0.10, 0.28)


class RetrieveFromBasketTests(unittest.TestCase):
    def test_place_then_retrieve_from_basket(self):
        """Right hand's counterpart for this rig: one arm places the object in the
        basket, then grasps it back out from inside the walls (not a bare-table
        pick relabelled) and sets it down elsewhere. Both legs run in the same
        MjModel/MjData (Demo(scene=...)), so the second grasp is really contending
        with the basket the first leg actually built."""
        task = RetrieveDemo(PICK_A, BASKET_MIDDLE, RETRIEVE_TO, side="left")
        result = run_retrieve_trial(task)
        self.assertTrue(result.success, result.failure_reason)
        self.assertLess(result.placement_error_m, 0.05)
        self.assertLess(result.bottle_tilt_deg, 5.0)
        # A successful result already implies a real antipodal pinch during the
        # grasp (demo.py's fingers_not_pressing gate, checked before proof lift) --
        # by the time the trial returns the hand has released and retreated, so
        # finger_contact_forces() here would read back near zero, not the grip.
        self.assertGreaterEqual(result.grasp_forces["thumb"], 6.0)

    def test_place_leg_failure_is_not_masked_as_a_retrieve_failure(self):
        """An unreachable pick position must fail in the place-into-basket leg
        with its own reason, not be silently swallowed or blamed on retrieval."""
        task = RetrieveDemo((0.60, -0.60), BASKET_MIDDLE, RETRIEVE_TO, side="left")
        result = run_retrieve_trial(task)
        self.assertFalse(result.success)
        self.assertIn("place-into-basket leg", result.failure_reason)


if __name__ == "__main__":
    unittest.main()
