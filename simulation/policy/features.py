"""Observation and action features of the grasp policy -- one definition shared by the
training data (scripts/train_grasp_policy.py, from the .npz demos) and the controller
running in MuJoCo (simulation/pick_place/policy_demo.py), so the two cannot drift.

observation (38):
    robot (27)   arm joints 7, hand joints 6, finger forces 5 (/ 20 N),
                 wrist position 3 (world), wrist rotation 6 (rot6d, world)
    object (11)  estimated object position in the wrist frame 3,
                 object symmetry axis in the wrist frame 3 (0 for round fruit),
                 kind one-hot 5 (can upright, can lying, apple, orange, peach)
action per step (15):
    commanded wrist position relative to the measured wrist, in the wrist frame 3,
    commanded wrist rotation relative to the measured one 6 (rot6d),
    hand actuator commands 6
"""

from __future__ import annotations

import numpy as np

KINDS = (("can", "upright"), ("can", "lying"), ("apple", "upright"), ("orange", "upright"), ("peach", "upright"))
ROUND = {"apple", "orange", "peach"}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")  # as in the demos
FORCE_SCALE_N = 20.0
OBS_DIM = 38
ACTION_DIM = 15


def rot6d(rotation: np.ndarray) -> np.ndarray:
    return np.asarray(rotation)[..., :, :2].swapaxes(-1, -2).reshape(*np.shape(rotation)[:-2], 6)


def from_rot6d(values: np.ndarray) -> np.ndarray:
    """Gram-Schmidt back to a rotation matrix."""
    a, b = np.asarray(values[:3], dtype=float), np.asarray(values[3:6], dtype=float)
    x = a / max(np.linalg.norm(a), 1e-9)
    b = b - x * (x @ b)
    y = b / max(np.linalg.norm(b), 1e-9)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def kind_onehot(kind: str, pose: str) -> np.ndarray:
    out = np.zeros(len(KINDS))
    out[KINDS.index((kind, pose))] = 1.0
    return out


def observation(arm_q, hand_q, forces, wrist: np.ndarray, object_est: np.ndarray, kind: str, pose: str) -> np.ndarray:
    """One observation vector; `wrist`, `object_est`: 4x4 world poses."""
    r = wrist[:3, :3]
    rel = r.T @ (object_est[:3, 3] - wrist[:3, 3])
    axis = np.zeros(3) if kind in ROUND else r.T @ object_est[:3, 2]
    return np.concatenate([arm_q, hand_q, np.asarray(forces) / FORCE_SCALE_N, wrist[:3, 3], rot6d(r),
                           rel, axis, kind_onehot(kind, pose)]).astype(np.float32)


def action(wrist: np.ndarray, wrist_cmd: np.ndarray, hand_ctrl) -> np.ndarray:
    r = wrist[:3, :3]
    return np.concatenate([r.T @ (wrist_cmd[:3, 3] - wrist[:3, 3]), rot6d(r.T @ wrist_cmd[:3, :3]),
                           hand_ctrl]).astype(np.float32)


def apply_action(wrist: np.ndarray, act: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(target wrist 4x4 world, hand commands) from an action and the measured wrist."""
    r = wrist[:3, :3]
    target = np.eye(4)
    target[:3, 3] = wrist[:3, 3] + r @ act[:3]
    target[:3, :3] = r @ from_rot6d(act[3:9])
    return target, np.asarray(act[9:15], dtype=float)


def episode_arrays(demo) -> tuple[np.ndarray, np.ndarray]:
    """(observations (T, OBS_DIM), actions (T, ACTION_DIM)) of one saved demo (.npz)."""
    kind, pose = str(demo["kind"]), str(demo["pose"])
    obs = np.stack([observation(demo["arm_q"][t], demo["hand_q"][t], demo["forces"][t], demo["wrist"][t],
                                demo["object_est"][t], kind, pose) for t in range(len(demo["time"]))])
    act = np.stack([action(demo["wrist"][t], demo["wrist_cmd"][t], demo["hand_ctrl"][t]) for t in range(len(demo["time"]))])
    return obs, act
