import unittest
from unittest.mock import patch

import numpy as np

from simulation.pick_place.demo import Demo, run_trial
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
        """Both hand-off strategies were audited and neither works for THIS specific
        opposite-workspace layout (pick deep in the right arm's own zone, basket
        deep in the left's): simultaneous bimanual grasp collides
        (scripts/check_handoff_geometry.py, >=22.7mm inter-hand penetration in every
        sampled pose) and a sequential relay has no reachable staging point at this
        object/basket pair either.

        This is *not* a blanket "can't grasp near the centreline" limit -- a direct
        single-arm pick from a moderate starting point (not this test's extreme
        (0.08, -0.38)) can grasp and place exactly on the centreline
        (test_delivers_object_to_true_centreline_basket below, and the
        pre-existing artifacts/episodes/centre-to-right recording: object grasped
        at (0.22, 0.0) with grasp_yaw=90). What genuinely has no solution is *this*
        pick/basket pair specifically: relaying it through any staging point still
        needs a transfer path crossing almost the whole table, and every waypoint
        on that path was checked and found unreachable (docs/PROJECT_REPORT.md)."""
        decision = TaskRouter(Scene((0.08, -0.38), (0.25, 0.25))).select()
        self.assertEqual(decision.route, Route.REJECTED)
        self.assertIn("handoff", decision.reason)

    def test_delivers_object_to_true_centreline_basket(self):
        """Straight in front of the torso (y=0), not just close to it: matches the
        already-recorded artifacts/episodes/centre-to-right (object grasped AT
        (0.22, 0.0)) run the other way round -- pick from a normal spot, carry to
        a basket sitting on the centreline. No planner change was needed; the
        earlier belief that x=0.30 or a very deep pick like PICK_POSITION_A ruled
        this out conflated "the transfer path from an extreme pick crosses the
        whole table" with "the centreline itself is unreachable". Reproduce with
        `python -m simulation.pick_place_demo --object 0.10 -0.35 --basket 0.22 0.0`."""
        demo = Demo((0.10, -0.35), (0.22, 0.0), side="right")
        plan = demo.planner.plan(demo.object_position())
        self.assertIn("transfer", plan.paths)
        result = run_trial(demo)
        self.assertTrue(result.success, result.failure_reason)
        self.assertLess(result.placement_error_m, 0.03)
        self.assertLess(result.bottle_tilt_deg, 5.0)

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
        """A grasp at a joint limit must be rejected even when IK solves it. The
        original layout here ((0.11024, -0.29146) -> (0.24123, -0.17028)) had its
        grasp margin pinned at 0.0deg on the pre-2026-09-23 geometry; after the
        transfer multi-seed fix it plans with a healthy 5.9deg grasp margin and
        executes in physics, so the guard moved to a layout whose grasp genuinely
        sits at the limit: yaw +30 on (0.14, -0.27) fails 'grasp joint margin only
        1.1deg' (probed via plan_pick, which tries every heading -- the heading
        filter below pins the plan to that one)."""
        from unittest.mock import patch

        from simulation.pick_place import config as C

        scene = Scene((0.14, -0.27), (0.26, -0.14))
        planner = GraspPlanner(scene)
        with patch.object(C, "GRASP_YAW_CANDIDATES_DEG", (30.0,)):
            with self.assertRaisesRegex(RuntimeError, "grasp joint margin only"):
                planner.plan(scene.object_position())


if __name__ == "__main__":
    unittest.main()
