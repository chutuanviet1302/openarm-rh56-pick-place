#!/usr/bin/env bash
# Build nvdiffrast's CUDA plugin (first use) and import FoundationPose's estimator.
set -eo pipefail
source "${CONDA_HOME:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate foundationpose
export CUDA_HOME="$CONDA_PREFIX" MAX_JOBS="${MAX_JOBS:-1}"
cd "${FP_HOME:-$HOME/foundationpose}/FoundationPose"
python - <<'PY'
import nvdiffrast.torch as dr, torch
ctx = dr.RasterizeCudaContext(); print("nvdiffrast OK")
import estimater; print("estimater import OK")
from estimater import ScorePredictor, PoseRefinePredictor
s, r = ScorePredictor(), PoseRefinePredictor(); print("weights loaded OK")
PY
