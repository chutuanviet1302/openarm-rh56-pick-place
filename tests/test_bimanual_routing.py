import unittest
from unittest.mock import patch

import numpy as np

from simulation.pick_place.demo import Demo
from simulation.pick_place.planner import GraspPlanner, Plan
from simulation.pick_place.routing import Route, TaskRouter
from simulation.pick_place.scene import Scene
from simulation.vision_detector import VisionDetector


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

    def test_head_camera_sees_mirrored_left_workspace(self):
        scene = Scene((0.08, 0.38), (0.25, 0.25))
        detection = VisionDetector(scene.model, "d435_head").detect_object(scene.data, render_annotation=False)
        self.assertTrue(detection.found)
        self.assertLess(float(np.linalg.norm(detection.pos_world[:2] - scene.object_position()[:2])), 0.01)

    def test_invalid_side_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "side"):
            Demo(side="centre")

    def test_planner_retries_grasp_heading_when_set_down_cannot_plan(self):
        planner = GraspPlanner(Scene())
        attempts = []

        def place(plan, _):
            attempts.append(plan.grasp_yaw_deg)
            if len(attempts) == 1:
                raise RuntimeError("set-down unreachable")
            return plan

        with patch.object(planner, "_plan_pick", side_effect=lambda _: Plan({})), patch.object(
            planner, "plan_place", side_effect=place
        ):
            plan = planner.plan(np.zeros(3))
        self.assertEqual(attempts, [0.0, -30.0])
        self.assertEqual(plan.grasp_yaw_deg, -30.0)

    def test_grasp_at_joint_limit_is_not_an_executable_layout(self):
        scene = Scene((0.11024, -0.29146), (0.24123, -0.17028))
        with self.assertRaisesRegex(RuntimeError, "grasp joint margin only"):
            GraspPlanner(scene).plan(scene.object_position())


if __name__ == "__main__":
    unittest.main()
