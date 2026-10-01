"""Convert the scripted grasp demos (artifacts/demos/*.npz) to a LeRobotDataset.

    .venv-lerobot/Scripts/python -m scripts.to_lerobot --demos artifacts/demos --root artifacts/lerobot/openarm_grasp

Features (simulation/policy/features.py, the single definition shared with the runtime):
    observation.state              (27)  arm + hand joints, finger forces, wrist pose
    observation.environment_state  (11)  perceived object in the wrist frame, kind one-hot
    action                         (15)  wrist target relative to the wrist, hand commands
State-only: no images, which LeRobot's ACT / Diffusion accept when the environment
state is given. 10 fps, one task string per object kind.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from simulation.policy.features import ACTION_DIM, OBS_DIM, episode_arrays

ROBOT_DIM = 27
FPS = 10


def main(argv: list[str] | None = None) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=Path, default=Path("artifacts") / "demos")
    parser.add_argument("--root", type=Path, default=Path("artifacts") / "lerobot" / "openarm_grasp")
    parser.add_argument("--repo-id", default="local/openarm_grasp")
    args = parser.parse_args(argv)
    if args.root.exists():
        shutil.rmtree(args.root)
    features = {
        "observation.state": {"dtype": "float32", "shape": (ROBOT_DIM,), "names": None},
        "observation.environment_state": {"dtype": "float32", "shape": (OBS_DIM - ROBOT_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }
    dataset = LeRobotDataset.create(args.repo_id, FPS, features, robot_type="openarm_rh56", root=args.root, use_videos=False)
    files = sorted(args.demos.glob("*.npz"))
    for path in files:
        demo = np.load(path)
        obs, act = episode_arrays(demo)
        task = f"pick up the {demo['kind']} ({demo['pose']})"
        for t in range(len(obs)):
            dataset.add_frame({
                "observation.state": obs[t, :ROBOT_DIM],
                "observation.environment_state": obs[t, ROBOT_DIM:],
                "action": act[t],
                "task": task,
            })
        dataset.save_episode()
    dataset.finalize()
    print(f"{len(files)} episodes, {dataset.num_frames} frames -> {args.root}")


if __name__ == "__main__":
    main()
