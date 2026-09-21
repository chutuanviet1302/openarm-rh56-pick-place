import unittest

from simulation.pick_place.demo import Demo
from simulation.pick_place.routing import Route, TaskRouter
from simulation.pick_place.scene import Scene


class BimanualRoutingTests(unittest.TestCase):
    def test_mirrored_layout_plans_with_left_arm(self):
        demo = Demo((0.08, 0.38), (0.25, 0.25), side="left")
        plan = demo.planner.plan(demo.object_position())
        self.assertEqual(plan.side, "left")
        self.assertIn("transfer", plan.paths)

    def test_router_uses_right_direct_route_for_default_layout(self):
        decision = TaskRouter(Scene()).select()
        self.assertEqual(decision.route, Route.DIRECT_RIGHT)

    def test_router_requests_handoff_for_opposite_workspaces(self):
        decision = TaskRouter(Scene((0.08, -0.38), (0.25, 0.25))).select()
        self.assertEqual(decision.route, Route.HANDOFF_RIGHT_TO_LEFT)

    def test_invalid_side_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "side"):
            Demo(side="centre")


if __name__ == "__main__":
    unittest.main()
