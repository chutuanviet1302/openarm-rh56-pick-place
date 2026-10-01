"""Train the action-chunking grasp policy on the scripted demos (behaviour cloning).

    python -m scripts.train_grasp_policy --demos artifacts/demos --out artifacts/policy/grasp_policy.pt

Episodes are split train / validation by file (10 % held out); the loss is L1 on the
normalised action chunk (ACT's choice: sharper than L2 for multi-modal demos). A chunk
that runs past the end of an episode is padded with the episode's last action (the
hand keeps holding) and those steps still count. CPU is enough for ~40-d state inputs;
on Kaggle the same script runs on the GPU (--device cuda).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from simulation.policy.features import episode_arrays
from simulation.policy.model import CHUNK, ChunkPolicy


def load_episodes(directory: Path) -> list[tuple[np.ndarray, np.ndarray, str]]:
    episodes = []
    for path in sorted(directory.glob("*.npz")):
        demo = np.load(path)
        obs, act = episode_arrays(demo)
        episodes.append((obs, act, path.name))
    return episodes


def chunks(episodes, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    obs, act = [], []
    for o, a, _ in episodes:
        padded = np.concatenate([a, np.repeat(a[-1:], chunk, axis=0)])
        for t in range(len(o)):
            obs.append(o[t])
            act.append(padded[t:t + chunk])
    return np.asarray(obs, dtype=np.float32), np.asarray(act, dtype=np.float32)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", type=Path, default=Path("artifacts") / "demos")
    parser.add_argument("--out", type=Path, default=Path("artifacts") / "policy" / "grasp_policy.pt")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    episodes = load_episodes(args.demos)
    order = rng.permutation(len(episodes))
    n_val = max(1, len(episodes) // 10)
    val = [episodes[i] for i in order[:n_val]]
    train = [episodes[i] for i in order[n_val:]]
    obs, act = chunks(train, CHUNK)
    vobs, vact = chunks(val, CHUNK)
    print(f"{len(train)} train / {len(val)} val episodes, {len(obs)} / {len(vobs)} samples, device {args.device}")

    policy = ChunkPolicy()
    policy.set_normalisation(obs, act.reshape(-1, act.shape[-1]))
    policy.to(args.device)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.steps)
    obs_t, act_t = torch.as_tensor(obs, device=args.device), torch.as_tensor(act, device=args.device)
    vobs_t, vact_t = torch.as_tensor(vobs, device=args.device), torch.as_tensor(vact, device=args.device)

    def loss_of(o, a) -> torch.Tensor:
        target = (a - policy.act_mean) / policy.act_std
        return torch.nn.functional.l1_loss(policy(o), target)

    best, history, started = float("inf"), [], time.perf_counter()
    for step in range(1, args.steps + 1):
        policy.train()
        index = torch.randint(len(obs_t), (args.batch,), device=args.device)
        loss = loss_of(obs_t[index], act_t[index])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()
        if step % 500 == 0 or step == args.steps:
            policy.eval()
            with torch.no_grad():
                val_loss = float(loss_of(vobs_t, vact_t))
            history.append({"step": step, "train": float(loss), "val": val_loss})
            print(f"step {step:6d}  train {float(loss):.4f}  val {val_loss:.4f}  ({time.perf_counter() - started:.0f}s)", flush=True)
            if val_loss < best:
                best = val_loss
                policy.to("cpu").save(args.out, step=step, val_l1=val_loss, episodes=len(episodes),
                                      val_files=[name for _, _, name in val])
                policy.to(args.device)
    args.out.with_suffix(".json").write_text(json.dumps({"best_val_l1": best, "history": history}, indent=2))
    print(f"best val L1 {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
