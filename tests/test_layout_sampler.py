import unittest
from unittest import mock

import numpy as np

from simulation.five_finger_model import BASKET_WALL_THICKNESS
from simulation.objects import footprint_radius
from simulation.pick_place import config as C
from simulation.pick_place import layout as L


def _always_plans(layout, name, side, timeout=None):
    return True, "stub"


class LayoutSamplerTest(unittest.TestCase):
    def test_box_stands_on_the_platform_and_keeps_clear_of_the_belt(self):
        (cx, cy), (hx, hy) = L.derive_box((0.17, 0.67))
        self.assertAlmostEqual(cx - hx - BASKET_WALL_THICKNESS, 0.17)
        self.assertAlmostEqual(cx + hx + BASKET_WALL_THICKNESS, L.BELT.x - 0.5 * L.BELT.width - C.PATH_CLEARANCE)
        self.assertEqual(cy, 0.0)

    def test_drop_spots_keep_the_object_inside_on_the_arms_half_and_away_from_earlier_drops(self):
        half = (0.065, 0.20)
        r = 0.04
        spots = L.drop_spot_candidates(half, "right", r, dropped=[(0.0, -0.05)])
        self.assertTrue(all(abs(x) <= half[0] - r + 1e-9 and -half[1] + r - 1e-9 <= y <= 0.0 for x, y in spots))
        first = spots[0]
        self.assertGreaterEqual(np.hypot(first[0], first[1] + 0.05), 2 * r - 1e-9)  # free: no landing on the earlier drop
        left = L.drop_spot_candidates(half, "left", r, dropped=[])
        self.assertTrue(all(y >= 0.0 for _, y in left))
        self.assertEqual(left[0], (0.0, 0.1))  # nothing dropped yet: the centre of the arm's half

    def test_sampling_is_reproducible_and_keeps_objects_apart(self):
        with mock.patch.object(L, "plan_check_isolated", _always_plans):
            a = L.sample_layout(5, verbose=False)
            b = L.sample_layout(5, verbose=False)
            c = L.sample_layout(6, verbose=False)
        self.assertEqual(a.to_json(), b.to_json())
        self.assertNotEqual(a.to_json(), c.to_json())
        self.assertEqual(len(a.table), 4)
        self.assertEqual(len(a.belt), 2)
        everything = a.table + a.belt
        for i, p in enumerate(everything):
            for q in everything[i + 1:]:
                gap = np.hypot(p.xy[0] - q.xy[0], p.xy[1] - q.xy[1]) - footprint_radius(p) - footprint_radius(q)
                self.assertGreater(gap, 0.0, (p.label, q.label))
        # Table objects outside the box walls, belt objects out of the camera's belt view.
        for p in a.table:
            self.assertGreater(abs(p.xy[1]), a.box_half[1] + BASKET_WALL_THICKNESS)
        for p in a.belt:
            self.assertGreater(p.xy[1], 0.6)
        labels = [p.label for p in everything]
        self.assertEqual(len(labels), len(set(labels)))

    def test_hand_half_width_constants_match_the_model(self):
        from simulation.pick_place.scene import Scene

        scene = Scene()
        for side, value in L.HAND_HALF_WIDTH.items():
            self.assertAlmostEqual(L.hand_half_width(scene, side), value, delta=0.001)

    def test_json_round_trip(self):
        with mock.patch.object(L, "plan_check_isolated", _always_plans):
            layout = L.sample_layout(2, verbose=False)
        again = L.Layout.from_json(layout.to_json())
        self.assertEqual(again.to_json(), layout.to_json())


if __name__ == "__main__":
    unittest.main()
