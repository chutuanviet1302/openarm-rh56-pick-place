import ast
import inspect
import unittest

from simulation.five_finger_model import build_five_finger_model
from simulation.pick_place_demo import Demo, Executor


class PhysicsIntegrityTests(unittest.TestCase):
    def test_overhead_camera_and_opposite_side_targets(self):
        import mujoco
        import numpy as np
        model = build_five_finger_model(pick_bottle=True)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera = model.camera("overhead").id
        np.testing.assert_allclose(-data.cam_xmat[camera].reshape(3, 3)[:, 2], [0, 0, -1])
        pick = data.xpos[model.body("pick_bottle").id]
        place = data.xpos[model.body("place_basket").id]
        # Keep the approved layout; assert physical separation rather than a stale
        # assumption that the basket must be on the centre line.
        self.assertLess(pick[1], -0.30)
        self.assertGreaterEqual(np.linalg.norm(pick[:2] - place[:2]), 0.15)
        self.assertGreater(pick[2], 0.0)
        self.assertGreater(place[2], 0.0)

    def test_object_mass_is_not_increased_by_visual_mesh(self):
        model = build_five_finger_model(pick_bottle=True)
        self.assertAlmostEqual(float(model.body_mass[model.body("pick_bottle").id]), 0.2)

    def test_runtime_does_not_overwrite_physics_state(self):
        # Everything that steps time lives in Executor; Demo.run only sequences phases.
        # None of it may write the robot's or the object's qpos/qvel directly.
        for cls in (Executor, Demo):
            tree = ast.parse(inspect.getsource(cls))
            for method in tree.body[0].body:
                if not isinstance(method, ast.FunctionDef):
                    continue
                self._assert_no_state_writes(method)

    def _assert_no_state_writes(self, method):
        if True:
            for node in ast.walk(method):
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        text = ast.unparse(target)
                        self.assertNotIn("self.data.qpos", text)
                        self.assertNotIn("self.data.qvel", text)

    def test_thumb_opposition_is_required(self):
        self.assertTrue(Demo.REQUIRE_THUMB_OPPOSITION)

    def test_model_has_no_hand_object_weld(self):
        model = build_five_finger_model(pick_bottle=True)
        names = {model.equality(index).name for index in range(model.neq)}
        self.assertFalse(names & {"grasp_left_box", "grasp_right_box"})

    def test_attention_stance_is_clear_of_the_robot_and_its_limits(self):
        """The rest pose every episode starts and ends in must not lean on the robot's
        own pedestal/torso (the old all-zero pose pressed the RH56 thumb into
        robot_riser: 9.4Nm on the DM4310 wrist, rated 7Nm) nor sit on a joint limit
        (joint4 was exactly at 0deg)."""
        import mujoco
        import numpy as np

        from simulation.pick_place.scene import Scene

        wrist_limit_nm = 7.0  # DM4310, joints 5-7 (openarm_mujoco v1)
        for kwargs in ({}, dict(arm_half_separation=0.06, left_arm_mount_yaw_deg=-100.0, right_arm_mount_yaw_deg=40.0,
                                attention_deg={"left": (20.0, -10.0, 0.0, 10.0, 0.0, 0.0, 0.0)})):
            scene = Scene((0.26, -0.26), (0.32, -0.02), **kwargs)
            scene.reset()
            for _ in range(1500):
                mujoco.mj_step(scene.model, scene.data)
            self.assertEqual(scene.robot_body_contacts(), {})
            for side in ("left", "right"):
                for index, qpos in enumerate(scene.arm_qpos[side]):
                    joint = next(j for j in range(scene.model.njnt) if scene.model.jnt_qposadr[j] == qpos)
                    low, high = scene.model.jnt_range[joint]
                    margin = np.degrees(min(scene.data.qpos[qpos] - low, high - scene.data.qpos[qpos]))
                    self.assertGreater(margin, 5.0, f"{side} joint{index + 1} at rest")
                wrist = np.abs(scene.data.actuator_force[scene.arm_actuators[side]][4:])
                self.assertLess(wrist.max(), wrist_limit_nm, f"{side} wrist hold torque at rest")
