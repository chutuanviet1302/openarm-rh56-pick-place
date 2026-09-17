import unittest

import mujoco
import numpy as np

from simulation.openarm_mujoco import LEFT_ARM_ACTUATORS, LEFT_EE_SITE, MujocoRobot, official_model_path
from simulation.five_finger_model import HAND_PREFIX, build_five_finger_model
from simulation.pick_place_demo import Demo


class MujocoSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = MujocoRobot()
        cls.robot.reset()

    def test_official_scene_exists_and_loads(self):
        self.assertTrue(official_model_path().is_file())
        self.assertEqual(self.robot.model.nu, 16)

    def test_required_left_arm_controls_exist(self):
        for name in LEFT_ARM_ACTUATORS:
            self.assertGreaterEqual(self.robot.model.actuator(name).id, 0)
        self.assertGreaterEqual(self.robot.model.site(LEFT_EE_SITE).id, 0)

    def test_scene_steps_and_end_effector_pose_is_finite(self):
        self.robot.step(0.05)
        pose = self.robot.end_effector_pose()
        self.assertTrue(np.all(np.isfinite(pose.position)))
        np.testing.assert_allclose(pose.rotation.T @ pose.rotation, np.eye(3), atol=1e-6)

    def test_joint_target_rejects_values_outside_limits(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.robot.set_joint_targets(np.full(7, 100.0))

    def test_inspire_hands_replace_stock_grippers(self):
        robot = MujocoRobot(five_finger=True)
        model = robot.model
        names = [model.joint(index).name for index in range(model.njnt)]
        actuators = [model.actuator(index).name for index in range(model.nu)]
        self.assertEqual(sum(name.startswith(HAND_PREFIX) for name in names), 24)
        self.assertEqual(sum(name.startswith(HAND_PREFIX) for name in actuators), 12)
        self.assertNotIn("openarm_left_finger_joint1", names)
        self.assertNotIn("openarm_left_finger_joint2", names)
        self.assertNotIn("openarm_right_finger_joint1", names)
        self.assertNotIn("openarm_right_finger_joint2", names)
        hand_limits = model.actuator_ctrlrange[robot._hand_actuators]
        robot.set_hand_targets(np.mean(hand_limits, axis=1))
        robot.step(0.05)
        self.assertTrue(np.all(np.isfinite(robot.data.qpos)))

    def test_hands_have_opposite_chirality(self):
        model = build_five_finger_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        left_thumb = data.site_xpos[model.site("inspire_left_left_thumb_tip").id]
        left_middle = data.site_xpos[model.site("inspire_left_left_middle_tip").id]
        right_thumb = data.site_xpos[model.site("inspire_right_right_thumb_tip").id]
        right_middle = data.site_xpos[model.site("inspire_right_right_middle_tip").id]
        self.assertGreater(left_thumb[1] - left_middle[1], 0)
        self.assertLess(right_thumb[1] - right_middle[1], 0)

    def test_hands_attach_directly_to_wrist_without_broken_offset(self):
        demo = Demo()
        for side in ("left", "right"):
            wrist = demo.data.xpos[demo.model.body(f"openarm_{side}_ee_base_link").id]
            hand = demo.data.xpos[demo.model.body(f"inspire_{side}_base").id]
            self.assertLess(np.linalg.norm(wrist - hand), 0.005)

    def test_both_hands_point_down_in_attention_pose(self):
        # Demo() starts in a symmetric table-top stance: both hands over the table,
        # fists closed, ~7cm above table, palms facing robot body.
        demo = Demo()
        for side in ("left", "right"):
            base = demo.data.xpos[demo.model.body(f"inspire_{side}_base").id]
            # Both hands are positioned over the table (x > 0.15m)
            self.assertGreater(base[0], 0.15)
            # Palm normal faces towards robot body (-X)
            mat = demo.data.xmat[demo.model.body(f"inspire_{side}_base").id].reshape(3, 3)
            palm_normal = -mat[:, 0]
            self.assertLess(palm_normal[0], -0.70)
            # Hand clearance above table is ~7cm
            min_z = min(
                demo.data.geom_xpos[g, 2] - (demo.model.geom_size[g, 2] if demo.model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER) else demo.model.geom_size[g, 0])
                for g in range(demo.model.ngeom) if f"inspire_{side}" in (demo.model.body(demo.model.geom_bodyid[g]).name or "")
            )
            clearance = min_z - 0.40
            self.assertGreaterEqual(clearance, 0.05)
            self.assertLessEqual(clearance, 0.10)

    def test_robot_pedestal_and_bottle_are_on_table(self):
        model = build_five_finger_model(pick_bottle=True)
        table = model.geom("table_top")
        table_center = model.body(table.bodyid[0]).pos + table.pos
        self.assertLessEqual(table_center[0] - table.size[0], 0.0)
        self.assertGreater(model.body("openarm_left_base_link").pos[2], 0.40)
        self.assertEqual(model.joint("pick_bottle_joint").type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(model.geom("ycb_mustard_bottle_visual").type[0], mujoco.mjtGeom.mjGEOM_MESH)

    def test_pick_scene_has_colored_mustard_bottle_and_basket(self):
        model = build_five_finger_model(pick_bottle=True)
        geom = model.geom("ycb_mustard_bottle_visual")
        # The visual geom should have a material with a loaded texture
        # (the real Campbell's soup can label from P-161 YCB assets).
        mat_id = geom.matid[0]
        self.assertGreaterEqual(mat_id, 0, "visual geom must have a material")
        tex_id = model.mat_texid[mat_id][1]  # slot 1 = diffuse map
        self.assertGreaterEqual(tex_id, 0, "material must have a diffuse texture")
        self.assertGreater(model.tex_height[tex_id], 0, "texture must have non-zero height")
        for name in ("place_basket_bottom", "place_basket_left", "place_basket_right", "place_basket_front", "place_basket_back"):
            self.assertGreaterEqual(model.geom(name).id, 0)

    def test_right_grasp_requires_closed_fingers_and_two_contacts(self):
        demo = Demo()
        # Fingers open (regardless of contact count): not secure.
        demo.data.qpos[demo.hand_qpos["right"]] = demo.open_hand["right"]
        self.assertFalse(demo._right_grasp_is_secure())
        # Fingers closed but nothing nearby to contact: still not secure.
        demo.data.qpos[demo.hand_qpos["right"]] = demo.closed_hand["right"]
        mujoco.mj_forward(demo.model, demo.data)
        self.assertFalse(demo._right_grasp_is_secure())

    def test_demo_starts_in_symmetric_ready_pose_with_both_hands_closed(self):
        demo = Demo()
        left = demo.data.site_xpos[demo.ee_site_id["left"]]
        right = demo.data.site_xpos[demo.ee_site_id["right"]]
        np.testing.assert_allclose(left[[0, 2]], right[[0, 2]], atol=0.01)
        self.assertAlmostEqual(left[1], -right[1], delta=0.01)
        for side in ("left", "right"):
            np.testing.assert_allclose(demo.data.qpos[demo.hand_qpos[side]], demo.closed_hand[side])


if __name__ == "__main__":
    unittest.main()
