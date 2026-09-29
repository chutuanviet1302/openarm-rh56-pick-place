"""`lerobot-train` for Windows: identical, except the checkpoints/last symlink (which
needs admin / Developer Mode on Windows and crashed the run after saving) becomes a
text file naming the latest checkpoint.

    .venv-lerobot/Scripts/python -m scripts.lerobot_train_win --dataset.repo_id=... (same flags)
"""

from __future__ import annotations

from pathlib import Path

import lerobot.scripts.lerobot_train as lerobot_train
import lerobot.utils.train_utils as train_utils


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last = Path(checkpoint_dir).parent / "last.txt"
    last.write_text(Path(checkpoint_dir).name)
    return last


train_utils.update_last_checkpoint = update_last_checkpoint
lerobot_train.update_last_checkpoint = update_last_checkpoint

if __name__ == "__main__":
    lerobot_train.main()
