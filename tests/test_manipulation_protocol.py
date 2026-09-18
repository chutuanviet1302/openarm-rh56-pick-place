"""Manipulation-team protocol checks: perception accuracy, honest randomization and
the episode record. These are the guarantees the robot_manipulation skill relies on."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from simulation.pick_place_demo import (
    MIN_OBJECT_TO_BASKET_M,
    RANDOM_BASKET_BOX,
    RANDOM_PICK_BOX,
    Demo,
    TrialResult,
    object_to_basket_distance,
    run_trial,
    sample_layout,
)
from simulation.vision_detector import VisionDetector, fit_circle_known_radius


class PerceptionTests(unittest.TestCase):
    def test_circle_fit_recovers_centre_from_one_visible_side(self):
        centre, radius = np.array([0.3, -0.4]), 0.0354
        angles = np.linspace(np.pi * 0.6, np.pi * 1.4, 40)  # only the near-facing arc is visible
        points = centre + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        fitted = fit_circle_known_radius(points, radius, points.mean(axis=0))
        np.testing.assert_allclose(fitted, centre, atol=1e-4)

    def test_head_camera_locates_object_within_1cm_over_20_positions(self):
        demo = Demo()
        detector = VisionDetector(demo.model, "d435_head")
        rng = np.random.default_rng(0)
        errors = []
        for _ in range(20):
            demo.data.qpos[demo.bottle_qpos : demo.bottle_qpos + 2] = [
                rng.uniform(*RANDOM_PICK_BOX[0]),
                rng.uniform(*RANDOM_PICK_BOX[1]),
            ]
            mujoco.mj_forward(demo.model, demo.data)
            result = detector.detect_object(demo.data, render_annotation=False)
            self.assertTrue(result.found, "object visible on the floor but not detected")
            truth = demo.data.qpos[demo.bottle_qpos : demo.bottle_qpos + 3]
            errors.append(np.linalg.norm(result.pos_world[:2] - truth[:2]))
            # The floor-plane assumption fixes z; it must agree with where the can really is.
            self.assertAlmostEqual(float(result.pos_world[2]), float(truth[2]), delta=0.002)
        self.assertLess(max(errors), 0.01, f"worst xy error {max(errors)*1000:.1f}mm")

    def test_detector_does_not_read_object_state(self):
        # The old detector "estimated" the position by copying data.xpos of the object.
        import inspect

        source = inspect.getsource(VisionDetector.detect_object)
        self.assertNotIn('body("pick_bottle")', source)
        self.assertNotIn("xpos[obj_id]", source)

    def test_perception_failure_is_an_error_not_a_fallback(self):
        demo = Demo(perception=True)
        # Hide the object under the floor so the camera cannot see it.
        demo.data.qpos[demo.bottle_qpos + 2] = -1.0
        mujoco.mj_forward(demo.model, demo.data)
        with self.assertRaisesRegex(RuntimeError, "perception failed"):
            demo.perceive_object()


class RandomizationTests(unittest.TestCase):
    def test_object_to_basket_distance_measures_to_the_wall(self):
        from simulation.five_finger_model import BASKET_HALF_WIDTH, BASKET_WALL_THICKNESS

        outer = BASKET_HALF_WIDTH + BASKET_WALL_THICKNESS
        self.assertAlmostEqual(object_to_basket_distance((0.3, -0.4), (0.3, -0.2)), 0.2 - outer)
        self.assertEqual(object_to_basket_distance((0.3, -0.2), (0.3, -0.2)), 0.0)

    def test_sampled_layout_is_executable_and_inside_the_boxes(self):
        layout = sample_layout(np.random.default_rng(3))
        pick, basket = layout["pick_position"], layout["basket_position"]
        self.assertTrue(RANDOM_PICK_BOX[0][0] <= pick[0] <= RANDOM_PICK_BOX[0][1])
        self.assertTrue(RANDOM_PICK_BOX[1][0] <= pick[1] <= RANDOM_PICK_BOX[1][1])
        self.assertTrue(RANDOM_BASKET_BOX[0][0] <= basket[0] <= RANDOM_BASKET_BOX[0][1])
        self.assertTrue(RANDOM_BASKET_BOX[1][0] <= basket[1] <= RANDOM_BASKET_BOX[1][1])
        self.assertGreaterEqual(np.hypot(basket[0] - pick[0], basket[1] - pick[1]), 0.15)
        self.assertGreaterEqual(object_to_basket_distance(pick, basket), MIN_OBJECT_TO_BASKET_M)
        # Executable: the full waypoint chain solves on the sampled layout.
        demo = Demo(pick, basket)
        demo._solve_poses()
        demo.planner.validate(demo.plan)
        self.assertGreaterEqual(demo.planner.fingertip_floor_clearance(demo.plan["grasp"]), 0.005)


class EpisodeRecordTests(unittest.TestCase):
    def test_trial_records_full_evidence(self):
        result = run_trial(Demo(perception=True))
        self.assertIsInstance(result, TrialResult)
        self.assertTrue(result.success, result.failure_reason)
        self.assertTrue(result.perception_used)
        self.assertLess(result.perception_error_m, 0.01)
        self.assertEqual(set(result.grasp_forces), {"thumb", "index", "middle", "ring", "pinky"})
        self.assertGreaterEqual(result.grasp_forces["thumb"], 0.5)
        self.assertGreaterEqual(
            sum(result.grasp_forces[name] >= 0.5 for name in ("index", "middle", "ring", "pinky")), 2
        )
        self.assertLess(result.wrist_pitch_at_grasp_deg, 45.0)
        self.assertGreaterEqual(result.carry_clearance_above_rim_m, 0.045)
        self.assertLessEqual(result.proof_lift_hand_rise_m - result.proof_lift_rise_m, 0.01)
        self.assertLessEqual(result.placement_error_m, 0.02)
        self.assertLessEqual(result.bottle_tilt_deg, 15.0)
        for phase in ("hover", "pregrasp", "grasp", "lift", "transfer", "lower"):
            self.assertIn(phase, result.phase_wrist_positions)
            self.assertEqual(len(result.phase_joint_targets[phase]), 7)


if __name__ == "__main__":
    unittest.main()
