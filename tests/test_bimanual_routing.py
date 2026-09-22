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

    def test_router_rejects_opposite_workspaces_no_staging_point_reachable(self):
        """Both hand-off strategies were audited and neither works on this rig:
        simultaneous bimanual grasp collides (scripts/check_handoff_geometry.py,
        >=22.7mm inter-hand penetration in every sampled pose) and a sequential
        relay has no staging point either, because grasping and placing have very
        different reach envelopes -- a held object can be *placed* to within about
        2cm of the centreline, but freshly *grasping* one needs roughly 30cm of
        clearance from it (the arm runs out of joint margin closer in, independent
        of which yaw is tried; see docs/PROJECT_REPORT.md). No table point is both
        reachable as a placement for the source arm and a pick-up for the target,
        so the router correctly falls back to REJECTED instead of proposing a
        handoff it cannot execute."""
        decision = TaskRouter(Scene((0.08, -0.38), (0.25, 0.25))).select()
        self.assertEqual(decision.route, Route.REJECTED)
        self.assertIn("handoff", decision.reason)

    def test_find_handoff_point_returns_first_candidate_reachable_by_both_arms(self):
        """No layout in the sampled workspace actually has a usable staging point
        (see the REJECTED test above), so this exercises the search itself --
        first-candidate-wins over the (x, y) grid, both legs checked -- against a
        stubbed reachability check rather than a real, currently nonexistent,
        physical instance."""
        from simulation.pick_place import handoff as handoff_module

        seen = []

        def fake_reachable(pick_position, place_position, side):
            seen.append((pick_position, place_position, side))
            return True

        with patch.object(handoff_module, "_leg_reachable", side_effect=fake_reachable):
            staging = handoff_module.find_handoff_point((0.08, -0.38), (0.25, 0.25), "right", "left")
        self.assertAlmostEqual(staging[0], handoff_module.HANDOFF_X_RANGE[0])
        self.assertAlmostEqual(staging[1], handoff_module.HANDOFF_Y_RANGE[0])
        self.assertEqual(seen[0], ((0.08, -0.38), staging, "right"))
        self.assertEqual(seen[1], (staging, (0.25, 0.25), "left"))

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

        def place(plan, _, **kwargs):
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
