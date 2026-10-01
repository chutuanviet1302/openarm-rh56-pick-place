#!/usr/bin/env bash
# Install NVlabs/FoundationPose in WSL2 Ubuntu 22.04 (conda route, no Docker).
#
#   wsl -d Ubuntu-22.04 -- bash "/mnt/d/Program Files/Downloads/Vin Dynamics/Projects/wsl/setup_foundationpose.sh"
#
# Target: GTX 1650 (4 GB), Windows driver >= 520 (CUDA 11.8 runtime inside WSL).
# Downloads ~6-8 GB (Miniconda, CUDA toolkit 11.8, torch cu118, pytorch3d, repo).
# Each step is skipped when already done, so the script can be re-run after a failure.
# Prerequisite, once, with sudo (the script does not ask for a password):
#   sudo apt-get update && sudo apt-get install -y build-essential git wget libgl1 libglib2.0-0 libeigen3-dev
set -euo pipefail

FP_HOME="${FP_HOME:-$HOME/foundationpose}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
ENV_NAME=foundationpose
mkdir -p "$FP_HOME"

for tool in gcc g++ git wget; do
  command -v "$tool" >/dev/null || { echo "missing $tool: run the sudo apt-get line at the top first"; exit 1; }
done

echo "== 1/6 Miniconda"
if [ ! -x "$CONDA_HOME/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_HOME"
fi
# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"
# conda-forge / nvidia only: Anaconda's default channels need their Terms of Service
# accepted, which this script does not do on anyone's behalf.
conda config --remove channels defaults 2>/dev/null || true
conda config --add channels conda-forge
conda config --set channel_priority flexible

echo "== 2/6 conda env ($ENV_NAME, python 3.9, CUDA toolkit 11.8 for nvdiffrast/extension builds)"
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.9
fi
conda activate "$ENV_NAME"
if ! command -v nvcc >/dev/null; then
  conda install -y --override-channels -c "nvidia/label/cuda-11.8.0" -c conda-forge cuda-toolkit
  conda install -y --override-channels -c conda-forge eigen=3.4.0 cmake=3.26 ninja
fi
export CUDA_HOME="$CONDA_PREFIX"
# nvdiffrast's JIT build links -lcudart from $CUDA_HOME/lib64; conda keeps it in lib/.
[ -e "$CONDA_PREFIX/lib64" ] || ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"
# Its first-use compile needs RAM; one job keeps it inside WSL's memory limit.
export MAX_JOBS=1
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}:$CONDA_PREFIX/share/eigen3/cmake"

echo "== 3/6 torch 2.0.0 + cu118"
python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.0.0') else 1)" 2>/dev/null || \
  pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

echo "== 4/6 FoundationPose repo + python deps"
if [ ! -d "$FP_HOME/FoundationPose" ]; then
  git clone https://github.com/NVlabs/FoundationPose.git "$FP_HOME/FoundationPose"
fi
cd "$FP_HOME/FoundationPose"
# Network weights: copied once to $FP_HOME/weights (outside the repo, from the Google
# Drive download); the repo reads them from ./weights.
if [ -d "$FP_HOME/weights" ] && [ ! -e weights/2023-10-28-18-33-37 ]; then
  mkdir -p weights
  ln -sfn "$FP_HOME/weights/2023-10-28-18-33-37" weights/2023-10-28-18-33-37
  ln -sfn "$FP_HOME/weights/2024-01-11-20-02-45" weights/2024-01-11-20-02-45
fi
# requirements.txt pins torch too; filter it out so the cu118 build above stays.
grep -viE '^(torch|torchvision|torchaudio)([=<> ]|$)' requirements.txt > /tmp/fp_requirements.txt
pip install -r /tmp/fp_requirements.txt
# torch 2.0 is built against numpy 1.x (and opencv >= 4.11 wants numpy 2).
pip install "numpy<2" "opencv-python<4.11" "opencv-contrib-python<4.11"
# nvdiffrast builds its CUDA extension against the installed torch: no build isolation.
python -c "import nvdiffrast.torch" 2>/dev/null || \
  pip install --no-build-isolation --no-cache-dir "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"
# pytorch3d's prebuilt wheel comes from a --no-index link: its deps from PyPI first.
if ! python -c "import pytorch3d" 2>/dev/null; then
  pip install fvcore iopath
  pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
fi
# Kaolin is only needed for the model-free (reference-image) setup; we are model-based.

echo "== 5/6 C++/CUDA extensions (mycpp, mycuda)"
if ! ls mycpp/build/*.so >/dev/null 2>&1; then
  CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11" \
    bash build_all_conda.sh
fi

echo "== 6/6 check"
python - <<'PY'
import torch, nvdiffrast.torch as dr, pytorch3d
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"GPU memory free {free/2**30:.2f} / {total/2**30:.2f} GiB")
ctx = dr.RasterizeCudaContext()
print("nvdiffrast OK, pytorch3d", pytorch3d.__version__)
PY

cat <<EOF

Environment ready. Weights: $FP_HOME/weights (linked into the repo's weights/).
Optional, for the upstream demo: demo data mustard0 -> $FP_HOME/FoundationPose/demo_data/mustard0
Then run the smoke test:
  bash "/mnt/d/Program Files/Downloads/Vin Dynamics/Projects/wsl/fp_smoke_test.sh"
EOF
