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
        self.assertLess(pick[1], place[1])
        self.assertGreaterEqual(np.linalg.norm(pick[:2] - place[:2]), 0.15)

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
