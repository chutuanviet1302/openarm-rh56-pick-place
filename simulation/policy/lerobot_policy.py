"""Run a LeRobot policy (ACT / Diffusion, trained by lerobot-train on the dataset from
scripts/to_lerobot.py) as the grasp controller: 38-d observation in, 15-d action out.

    controller = LeRobotController("artifacts/lerobot_runs/act/checkpoints/040000/pretrained_model")
    action = controller(features.observation(...))

ACT runs with temporal ensembling (a new chunk every step, blended; Zhao et al. 2023).
The checkpoint's pre/post-processors (normalisation) are loaded with it; the device is
forced to CPU so a model trained on a Kaggle / Colab GPU runs on this laptop.
Needs the LeRobot environment (.venv-lerobot).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROBOT_DIM = 27
ACT_ENSEMBLE_COEFF = 0.01


class LeRobotController:
    def __init__(self, path: str | Path, device: str = "cpu") -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        path = str(path)
        config = PreTrainedConfig.from_pretrained(path)
        config.device = device
        if config.type == "act":
            config.temporal_ensemble_coeff = ACT_ENSEMBLE_COEFF
            config.n_action_steps = 1
        self.policy = get_policy_class(config.type).from_pretrained(path, config=config)
        self.policy.eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            config, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})
        self.policy.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        torch = self.torch
        batch = {
            "observation.state": torch.as_tensor(obs[:ROBOT_DIM], dtype=torch.float32),
            "observation.environment_state": torch.as_tensor(obs[ROBOT_DIM:], dtype=torch.float32),
        }
        with torch.no_grad():
            action = self.postprocess(self.policy.select_action(self.preprocess(batch)))
        return action.squeeze(0).cpu().numpy()
