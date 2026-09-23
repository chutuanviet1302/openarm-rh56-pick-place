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

    def test_right_places_left_retrieves(self):
        """Cross-arm: the right arm places into the basket, the left arm grasps it
        back out and sets it down on the table. On the stock mount the two arms'
        regions don't overlap (right places no further than y~0, left can only
        grasp well outboard of its shoulder), so this runs on a simulation-only
        layout: arm roots 0.10m off centre and the left mount turned -120deg about
        the vertical, swinging its grasp region in over the right arm's side.
        (With the stock 0.031m separation that turn put left link1 into right
        link0 and the grasp missed.)"""
        task = RetrieveDemo(
            (0.10, -0.35), (0.28, -0.14), (0.12, -0.22), side="left", place_side="right",
            arm_half_separation=0.10, left_arm_mount_yaw_deg=-120.0,
        )
        result = run_retrieve_trial(task)
        self.assertTrue(result.success, result.failure_reason)
        self.assertEqual((result.source_arm, result.target_arm), ("right", "left"))
        self.assertLess(result.placement_error_m, 0.05)
        self.assertLess(result.bottle_tilt_deg, 15.0)

    def test_right_places_left_retrieves_centre_basket(self):
        """Basket at the table centre (0.32, -0.02). Sim-only layout: arm roots 0.06m
        off centre, left mount turned -100deg and right +40deg about the vertical --
        the left arm's grasp window and the right arm's placing reach only meet
        at the centre with both turned. The executor aborts on any arm-arm contact
        and on the idle arm touching the object, so a pass means neither
        happened. The right hand's release leaves the can ~25mm short of the basket
        centre in x and ~12mm in -y (measured 2026-09-23); basket y=-0.02 and left
        mount -100deg put that landing spot inside the left arm's grasp window. The
        earlier (0.32, 0.0) / -95deg layout landed at x=0.295, 10mm outside it.
        Pick perturbed +-5mm/+-10mm: 6/7 pass (fails at pick x+10mm)."""
        task = RetrieveDemo(
            (0.26, -0.26), (0.32, -0.02), (0.34, 0.16), side="left", place_side="right",
            arm_half_separation=0.06, left_arm_mount_yaw_deg=-100.0, right_arm_mount_yaw_deg=40.0,
            # Left rest pose turned 20deg at the shoulder: the mirrored default hangs
            # the idle left hand in the basket on this rotated mount.
            attention_deg={"left": (20.0, -10.0, 0.0, 10.0, 0.0, 0.0, 0.0)},
        )
        result = run_retrieve_trial(task)
        self.assertTrue(result.success, result.failure_reason)
        self.assertEqual((result.source_arm, result.target_arm), ("right", "left"))
        self.assertLess(result.bottle_tilt_deg, 15.0)

    def test_place_leg_failure_is_not_masked_as_a_retrieve_failure(self):
        """An unreachable pick position must fail in the place-into-basket leg
        with its own reason, not be silently swallowed or blamed on retrieval."""
        task = RetrieveDemo((0.60, -0.60), BASKET_MIDDLE, RETRIEVE_TO, side="left")
        result = run_retrieve_trial(task)
        self.assertFalse(result.success)
        self.assertIn("place-into-basket leg", result.failure_reason)


if __name__ == "__main__":
    unittest.main()
