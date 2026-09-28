"""FoundationPose side of the bridge (runs in WSL, conda env `foundationpose`).

    python fp_run.py <frame_dir> [<frame_dir> ...]

Each frame_dir holds what simulation/fp_bridge.py exported: rgb.png, depth.npy (m),
mask.png, cam_K.txt, meta.json (mesh path). Writes T_cam_object.txt (4x4, OpenCV
camera frame, object = the OBJ's own frame) and fp_result.json (timing, GPU memory).

Models are loaded once per call, so several frames in one call pay the ~10 s start-up
once. GTX 1650 (4 GB): FP32 (FP_AMP=0: FP16 gave NaN poses on this card), a sparse
rotation grid (FP_VIEWS=12 x FP_INPLANE_STEP=90 deg instead of 40 x 60) and
FP_REFINE_ITER=3 (upstream 5).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

FP_DIR = Path(os.environ.get("FP_DIR", Path.home() / "foundationpose" / "FoundationPose"))
sys.path.insert(0, str(FP_DIR))
os.chdir(FP_DIR)  # the estimator finds its weights relative to the repo

import imageio.v3 as iio  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr  # noqa: E402

LOW_MEM = os.environ.get("FP_LOW_MEM", "1") == "1"
REFINE_ITER = int(os.environ.get("FP_REFINE_ITER", "3"))
AMP = os.environ.get("FP_AMP", "0") == "1"
# Rotation hypotheses (views x in-plane turns): every one is refined and scored in a
# single batch, so GPU memory grows with their number. FP32 on 4 GB: ~48 fit (12 x 4);
# the task objects are round or axially symmetric, which a sparse grid covers.
VIEWS = int(os.environ.get("FP_VIEWS", "12"))
INPLANE_STEP = int(os.environ.get("FP_INPLANE_STEP", "90"))


def main(frame_dirs: list[str]) -> None:
    started = time.perf_counter()
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    # GTX 16xx (Turing without tensor cores) returns NaN poses under FP16 autocast
    # (measured: every pose NaN on the GTX 1650); FP_AMP=1 turns it back on.
    scorer.amp = refiner.amp = AMP
    glctx = dr.RasterizeCudaContext()
    load_s = time.perf_counter() - started
    estimators: dict[str, FoundationPose] = {}
    for frame_dir in map(Path, frame_dirs):
        meta = json.loads((frame_dir / "meta.json").read_text())
        mesh_path = meta["mesh_wsl"]
        if mesh_path not in estimators:
            mesh = trimesh.load(mesh_path, force="mesh")
            estimator = FoundationPose(
                model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                scorer=scorer, refiner=refiner, debug=0, debug_dir=str(frame_dir), glctx=glctx,
            )
            if LOW_MEM:
                estimator.make_rotation_grid(min_n_views=VIEWS, inplane_step=INPLANE_STEP)
            estimators[mesh_path] = estimator
        estimator = estimators[mesh_path]
        rgb = iio.imread(frame_dir / "rgb.png")[..., :3]
        depth = np.load(frame_dir / "depth.npy").astype(np.float32)
        mask = iio.imread(frame_dir / "mask.png") > 0
        K = np.loadtxt(frame_dir / "cam_K.txt")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        pose = estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=REFINE_ITER)
        torch.cuda.synchronize()
        register_s = time.perf_counter() - t0
        np.savetxt(frame_dir / "T_cam_object.txt", np.asarray(pose, dtype=float).reshape(4, 4))
        result = {
            "object": meta.get("object"), "register_s": round(register_s, 3), "model_load_s": round(load_s, 2),
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20), "refine_iter": REFINE_ITER,
            "low_mem": LOW_MEM, "amp": AMP, "hypotheses": int(len(estimator.rot_grid)),
        }
        (frame_dir / "fp_result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
