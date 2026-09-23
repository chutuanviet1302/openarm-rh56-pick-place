import unittest

import mujoco
import numpy as np

from simulation.openarm_mujoco import LEFT_ARM_ACTUATORS, LEFT_EE_SITE, MujocoRobot, official_model_path
from simulation.five_finger_model import (
    TABLE_TOP_Z,
    HAND_PREFIX,
    ROBOT_RISER_HEIGHT,
    SHOULDER_AXIS_Z,
    build_five_finger_model,
)
from simulation.pick_place_demo import Demo
from simulation.pick_place.planner import GraspPlanner


class MujocoSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = MujocoRobot()
        cls.robot.reset()

    def test_official_scene_exists_and_loads(self):
        self.assertTrue(official_model_path().is_file())
        # OpenArm v1 bimanual: 7 + 7 arm servos and 2 + 2 stock gripper fingers.
        self.assertEqual(self.robot.model.nu, 18)

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

    def test_left_tip_sites_are_true_mirrors_not_folded_into_the_palm(self):
        demo = Demo()
        lengths = {}
        for side in ("left", "right"):
            planner = GraspPlanner(demo.scene, side)
            fingers, thumb = planner.local_jaw_offsets()
            lengths[side] = np.array([np.linalg.norm(fingers), np.linalg.norm(thumb)])
        np.testing.assert_allclose(lengths["left"], lengths["right"], atol=0.003)

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

    def test_attention_pose_does_not_touch_floor_or_riser(self):
        demo = Demo()
        model, data = demo.model, demo.data
        support_geoms = {model.geom(name).id for name in ("table_top", "robot_riser")}
        offenders = set()
        for contact in data.contact[: data.ncon]:
            if contact.dist >= 0 or not ({contact.geom1, contact.geom2} & support_geoms):
                continue
            other = contact.geom2 if contact.geom1 in support_geoms else contact.geom1
            body = model.body(model.geom_bodyid[other]).name or ""
            if "openarm" in body or "inspire" in body:
                offenders.add(body)
        self.assertFalse(offenders, f"robot touches the floor/riser in attention pose: {sorted(offenders)}")

    def test_openarm_v1_joint_and_servo_limits_match_vendor_model(self):
        model = build_five_finger_model()
        expected_deg = {
            "right": np.array([[-80, 200], [-10, 190], [-90, 90], [0, 140], [-90, 90], [-45, 45], [-90, 90]]),
            "left": np.array([[-200, 80], [-190, 10], [-90, 90], [0, 140], [-90, 90], [-45, 45], [-90, 90]]),
        }
        for side in ("left", "right"):
            joints = np.array([model.joint(f"openarm_{side}_joint{i}").range for i in range(1, 8)])
            controls = np.array([model.actuator(f"{side}_joint{i}_ctrl").ctrlrange for i in range(1, 8)])
            np.testing.assert_allclose(np.degrees(joints), expected_deg[side], atol=1e-3)
            np.testing.assert_array_equal(controls, joints)

    def test_hands_continue_the_forearm_axis(self):
        """The Inspire hand is bolted to the flange along the tool axis: its fingers
        point the way the forearm points (within a few degrees), the base sits on
        link7's flange face (+z, OpenArm v1) plus the adapter, and the palm faces the
        robot's midline with the arm hanging at rest."""
        from simulation.five_finger_model import HAND_MOUNT_Z

        model = build_five_finger_model(pick_bottle=True)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)  # all joints zero: arms hang straight down
        for side, inward in (("right", +1.0), ("left", -1.0)):
            flange = model.body(f"openarm_{side}_link7").id
            hand = model.body(f"inspire_{side}_base").id
            # Forearm axis = link5 origin -> link7 origin (v1's link6 is offset sideways
            # by the wrist gimbal, so link5 -> link6 is not the forearm direction).
            link5 = data.xpos[model.body(f"openarm_{side}_link5").id]
            link7 = data.xpos[flange]
            forearm = (link7 - link5) / np.linalg.norm(link7 - link5)
            tips = np.mean(
                [data.site_xpos[model.site(f"inspire_{side}_{side}_{f}_tip").id] for f in ("index", "middle", "ring", "pinky")],
                axis=0,
            )
            fingers = tips - data.xpos[hand]
            fingers /= np.linalg.norm(fingers)
            angle = np.degrees(np.arccos(np.clip(forearm @ fingers, -1.0, 1.0)))
            self.assertLess(angle, 5.0, f"{side} fingers are {angle:.1f} degrees off the forearm axis")
            # Base on the flange axis, HAND_MOUNT_Z along link7's +z (world down at rest).
            offset = data.xmat[flange].reshape(3, 3).T @ (data.xpos[hand] - data.xpos[flange])
            np.testing.assert_allclose(offset, [0.0, 0.0, HAND_MOUNT_Z], atol=1e-6)
            palm_normal = data.xmat[hand].reshape(3, 3)[:, 0]
            self.assertGreater(inward * palm_normal[1], 0.9, f"{side} palm should face the midline")

    def test_attention_pose_hangs_at_the_sides_palms_to_the_body(self):
        # Demo() starts at attention: arms straight down at the sides, fists closed,
        # fingers pointing down, palms facing the body, clear of the table top.
        demo = Demo()
        for side, inward in (("left", -1.0), ("right", +1.0)):
            base = demo.data.xpos[demo.model.body(f"inspire_{side}_base").id]
            # joint1 -20 / joint4 +40 at rest (config.ATTENTION_RIGHT) keeps the hand
            # ~3.7cm forward of the shoulder line.
            self.assertLess(abs(base[0]), 0.08)
            mat = demo.data.xmat[demo.model.body(f"inspire_{side}_base").id].reshape(3, 3)
            self.assertLess(mat[2, 2], -0.9, "fingers point down (-z)")
            self.assertGreater(inward * mat[1, 0], 0.9, "palm faces the body")
            min_z = min(
                demo.data.geom_xpos[g, 2] - (demo.model.geom_size[g, 2] if demo.model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER) else demo.model.geom_size[g, 0])
                for g in range(demo.model.ngeom) if f"inspire_{side}" in (demo.model.body(demo.model.geom_bodyid[g]).name or "")
            )
            self.assertGreaterEqual(min_z - TABLE_TOP_Z, 0.03)  # fingertips hang 4.0cm over the top

    def test_robot_riser_bottle_and_basket_are_on_the_table(self):
        model = build_five_finger_model(pick_bottle=True)
        for removed in ("table_pedestal_mount", "work_platform"):
            with self.assertRaises(KeyError):
                model.geom(removed)
        # The table top is the z=0 work surface and runs under the robot; the room
        # floor (vendor ground plane) is one measured table height below it.
        table = model.geom("table_top_visual")
        self.assertAlmostEqual(float(table.pos[2] + table.size[2]), TABLE_TOP_Z, places=3)
        self.assertLess(float(table.pos[0] - table.size[0]), -0.15)
        self.assertAlmostEqual(float(model.geom("table_top").pos[2]), TABLE_TOP_Z, places=6)
        self.assertAlmostEqual(float(model.geom("floor").pos[2]), TABLE_TOP_Z - 0.74, places=3)
        riser = model.geom("robot_riser")
        riser_center = model.body(riser.bodyid[0]).pos + riser.pos
        self.assertAlmostEqual(float(riser_center[2] - riser.size[2]), TABLE_TOP_Z, places=3)
        self.assertAlmostEqual(float(riser_center[2] + riser.size[2]), ROBOT_RISER_HEIGHT, places=3)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(float(data.xpos[model.body("openarm_left_link0").id][2]), SHOULDER_AXIS_Z, places=3)
        self.assertAlmostEqual(SHOULDER_AXIS_Z, 0.78 - 0.083, places=3)
        self.assertAlmostEqual(float(data.xpos[model.body("pick_bottle").id][2]), TABLE_TOP_Z + 0.05, places=3)
        basket = model.geom("place_basket_bottom")
        self.assertAlmostEqual(float(data.geom_xpos[basket.id][2] - basket.size[2]), TABLE_TOP_Z, places=3)
        basket_xy = data.geom_xpos[basket.id][:2]
        riser_xy = data.geom_xpos[riser.id][:2]
        self.assertTrue(np.any(np.abs(basket_xy - riser_xy) >= basket.size[:2] + riser.size[:2]))
        self.assertEqual(model.joint("pick_bottle_joint").type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(model.geom("ycb_mustard_bottle_visual").type[0], mujoco.mjtGeom.mjGEOM_MESH)

    def test_rejects_props_overlapping_robot_base(self):
        with self.assertRaisesRegex(ValueError, "pick object overlaps"):
            build_five_finger_model(pick_bottle=True, pick_position=(0.0, 0.0), basket_position=(0.30, -0.20))
        with self.assertRaisesRegex(ValueError, "basket overlaps"):
            build_five_finger_model(pick_bottle=True, pick_position=(0.08, -0.38), basket_position=(0.16, -0.10))

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
