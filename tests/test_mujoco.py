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
        """Left and right hands are mirror images across the sagittal (y=0) plane:
        with the arms hanging at rest, each thumb sits forward of its middle finger
        (+x, palms facing inward) and the thumbs' sideways offsets are opposite."""
        model = build_five_finger_model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        offsets = {}
        for side in ("left", "right"):
            thumb = data.site_xpos[model.site(f"inspire_{side}_{side}_thumb_tip").id]
            middle = data.site_xpos[model.site(f"inspire_{side}_{side}_middle_tip").id]
            offsets[side] = thumb - middle
        for side in ("left", "right"):
            self.assertGreater(offsets[side][0], 0.02, f"{side} thumb should be forward of the fingers")
        # Table-plane components mirror; z is left out because the vendor's left and
        # right hand models rest at different finger curls.
        mirrored = offsets["right"][:2] * np.array([1.0, -1.0])
        np.testing.assert_allclose(offsets["left"][:2], mirrored, atol=0.02)

    def test_palm_collides_and_does_not_touch_the_arm(self):
        """Every hand geom takes part in collision (the hand must not be able to pass
        through the basket or table), and with the hand mounted along the flange axis
        the palm shell no longer intersects the arm in the rest or attention poses."""
        demo = Demo()
        model, data = demo.model, demo.data
        for side in ("left", "right"):
            base = model.body(f"inspire_{side}_base").id
            # The vendor model carries one visual-only shell (group 2) and eight
            # collision primitives (group 3) on the palm; the latter must all collide.
            palm_collision = [g for g in range(model.ngeom) if model.geom_bodyid[g] == base and model.geom_group[g] != 2]
            self.assertGreaterEqual(len(palm_collision), 4)
            self.assertTrue(all(model.geom_contype[g] != 0 for g in palm_collision), f"{side} palm collision disabled")
        for pose in ("attention", "zero"):
            if pose == "zero":
                data.qpos[:] = 0.0
            mujoco.mj_forward(model, data)
            for contact in data.contact[: data.ncon]:
                b1 = model.body(model.geom_bodyid[contact.geom1]).name or ""
                b2 = model.body(model.geom_bodyid[contact.geom2]).name or ""
                hand_vs_arm = ("inspire" in b1) != ("inspire" in b2) and ("openarm" in b1 or "openarm" in b2)
                self.assertFalse(hand_vs_arm and contact.dist < 0, f"{pose}: {b1} intersects {b2}")

    def test_arms_hanging_straight_down_do_not_touch_the_table(self):
        """Standing at attention with every arm joint at zero, both arms hang beside
        the pedestal tongue and nothing of the robot touches the table."""
        model = build_five_finger_model(pick_bottle=True)
        data = mujoco.MjData(model)
        for side in ("left", "right"):
            for name in (f"openarm_{side}_joint{i}" for i in range(1, 8)):
                data.qpos[model.joint(name).qposadr[0]] = 0.0
        mujoco.mj_forward(model, data)
        table_geoms = {model.geom("table_top").id, model.geom("table_pedestal_mount").id}
        offenders = set()
        for contact in data.contact[: data.ncon]:
            if contact.dist >= 0 or not ({contact.geom1, contact.geom2} & table_geoms):
                continue
            other = contact.geom2 if contact.geom1 in table_geoms else contact.geom1
            body = model.body(model.geom_bodyid[other]).name or ""
            if "openarm" in body or "inspire" in body:
                offenders.add(body)
        self.assertFalse(offenders, f"robot touches the table with arms straight down: {sorted(offenders)}")

    def test_hands_continue_the_forearm_axis(self):
        """The Inspire hand is bolted to the flange along the tool axis: its fingers
        point the way the forearm points (within a few degrees), the base sits just
        past link6's shell on the flange axis, and the palm faces the robot's midline
        with the arm hanging at rest."""
        from simulation.five_finger_model import HAND_MOUNT_Z

        model = build_five_finger_model(pick_bottle=True)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)  # all joints zero: arms hang straight down
        for side, inward in (("right", +1.0), ("left", -1.0)):
            flange = model.body(f"openarm_{side}_ee_base_link").id
            hand = model.body(f"inspire_{side}_base").id
            link5 = data.xpos[model.body(f"openarm_{side}_link5").id]
            link6 = data.xpos[model.body(f"openarm_{side}_link6").id]
            forearm = (link6 - link5) / np.linalg.norm(link6 - link5)
            tips = np.mean(
                [data.site_xpos[model.site(f"inspire_{side}_{side}_{f}_tip").id] for f in ("index", "middle", "ring", "pinky")],
                axis=0,
            )
            fingers = tips - data.xpos[hand]
            fingers /= np.linalg.norm(fingers)
            angle = np.degrees(np.arccos(np.clip(forearm @ fingers, -1.0, 1.0)))
            self.assertLess(angle, 5.0, f"{side} fingers are {angle:.1f} degrees off the forearm axis")
            # Base on the flange axis, HAND_MOUNT_Z along the flange's -z (here world -z).
            offset = data.xmat[flange].reshape(3, 3).T @ (data.xpos[hand] - data.xpos[flange])
            np.testing.assert_allclose(offset, [0.0, 0.0, HAND_MOUNT_Z], atol=1e-6)
            palm_normal = data.xmat[hand].reshape(3, 3)[:, 0]
            self.assertGreater(inward * palm_normal[1], 0.9, f"{side} palm should face the midline")

    def test_attention_pose_holds_fists_forward_over_the_table(self):
        # Demo() starts in a symmetric stance: fists closed in front of the body over
        # the table, fingers forward, palms facing each other, wrist straight.
        demo = Demo()
        for side, inward in (("left", -1.0), ("right", +1.0)):
            base = demo.data.xpos[demo.model.body(f"inspire_{side}_base").id]
            self.assertGreater(base[0], 0.12)
            mat = demo.data.xmat[demo.model.body(f"inspire_{side}_base").id].reshape(3, 3)
            self.assertGreater(mat[0, 2], 0.9, "fingers point forward (+x)")
            self.assertGreater(inward * mat[1, 0], 0.7, "palm faces the midline")
            for index in (5, 6):
                self.assertAlmostEqual(float(demo.data.qpos[demo.arm_qpos[side][index]]), 0.0, places=3)
            # Hand clearance above table is ~7cm
            min_z = min(
                demo.data.geom_xpos[g, 2] - (demo.model.geom_size[g, 2] if demo.model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER) else demo.model.geom_size[g, 0])
                for g in range(demo.model.ngeom) if f"inspire_{side}" in (demo.model.body(demo.model.geom_bodyid[g]).name or "")
            )
            clearance = min_z - 0.40
            self.assertGreaterEqual(clearance, 0.05)
            self.assertLessEqual(clearance, 0.20)

    def test_robot_pedestal_and_bottle_are_on_table(self):
        model = build_five_finger_model(pick_bottle=True)
        # The pedestal's base block (x -0.155..0.095) stands on the rear tongue of the
        # table, and the work surface starts in front of the hanging arms.
        tongue = model.geom("table_pedestal_mount")
        tongue_center = model.body(tongue.bodyid[0]).pos + tongue.pos
        self.assertLessEqual(tongue_center[0] - tongue.size[0], -0.155)
        self.assertGreaterEqual(tongue_center[0] + tongue.size[0], 0.095)
        self.assertAlmostEqual(float(tongue_center[2] + tongue.size[2]), 0.40, places=3)
        table = model.geom("table_top")
        table_center = model.body(table.bodyid[0]).pos + table.pos
        self.assertGreaterEqual(table_center[0] - table.size[0], 0.12)
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
