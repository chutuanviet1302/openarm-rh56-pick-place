"""Action-chunking grasp policy (the ACT idea without the images): from one observation
predict the next CHUNK actions at once, and at run time blend the overlapping chunks
predicted at successive steps (temporal ensembling, Zhao et al. 2023). Chunking keeps a
policy trained by behaviour cloning from drifting off the demonstrated path one step at
a time; ensembling smooths the command.

Plain PyTorch (runs the same on this laptop's CPU and on a Kaggle GPU); normalisation
statistics are stored in the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from simulation.policy.features import ACTION_DIM, OBS_DIM

CHUNK = 10          # 1 s at 10 Hz
HIDDEN = 512
ENSEMBLE_DECAY = 0.1


class ChunkPolicy(nn.Module):
    def __init__(self, chunk: int = CHUNK, hidden: int = HIDDEN, dropout: float = 0.1) -> None:
        super().__init__()
        self.chunk = chunk
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, chunk * ACTION_DIM),
        )
        for name in ("obs_mean", "obs_std", "act_mean", "act_std"):
            size = OBS_DIM if name.startswith("obs") else ACTION_DIM
            self.register_buffer(name, torch.zeros(size) if name.endswith("mean") else torch.ones(size))

    def set_normalisation(self, obs: np.ndarray, act: np.ndarray) -> None:
        self.obs_mean.copy_(torch.as_tensor(obs.mean(0)))
        self.obs_std.copy_(torch.as_tensor(obs.std(0) + 1e-3))
        self.act_mean.copy_(torch.as_tensor(act.mean(0)))
        self.act_std.copy_(torch.as_tensor(act.std(0) + 1e-3))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalised action chunk (B, CHUNK, ACTION_DIM) from raw observations (B, OBS_DIM)."""
        x = (obs - self.obs_mean) / self.obs_std
        return self.net(x).view(-1, self.chunk, ACTION_DIM)

    def denormalise(self, act: torch.Tensor) -> torch.Tensor:
        return act * self.act_std + self.act_mean

    def save(self, path: Path, **meta) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.state_dict(), "chunk": self.chunk, "meta": meta}, path)

    @classmethod
    def load(cls, path: Path) -> "ChunkPolicy":
        data = torch.load(path, map_location="cpu")
        policy = cls(chunk=data["chunk"])
        policy.load_state_dict(data["state"])
        policy.eval()
        return policy


class EnsembledController:
    """Query the policy each step; the action for now is the exponentially weighted mean
    of every chunk that predicted this step (older predictions weigh more, as in ACT)."""

    def __init__(self, policy: ChunkPolicy) -> None:
        self.policy = policy
        self.history: list[tuple[int, np.ndarray]] = []   # (step predicted at, chunk)
        self.step = 0

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        chunk = self.policy.denormalise(self.policy(torch.as_tensor(obs)[None]))[0].numpy()
        self.history.append((self.step, chunk))
        self.history = [(s, c) for s, c in self.history if self.step - s < self.policy.chunk]
        preds = np.stack([c[self.step - s] for s, c in self.history])
        weights = np.exp(-ENSEMBLE_DECAY * np.arange(len(preds)))
        self.step += 1
        return (weights[:, None] * preds).sum(0) / weights.sum()
