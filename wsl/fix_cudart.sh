#!/usr/bin/env bash
# nvdiffrast's JIT build links -lcudart from $CUDA_HOME/lib64; conda's CUDA toolkit
# keeps its libraries under lib/ (or targets/x86_64-linux/lib). Point lib64 there.
set -euo pipefail
source "${CONDA_HOME:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate foundationpose
echo "env: $CONDA_PREFIX"
find "$CONDA_PREFIX" -name "libcudart.so*" | head -5
lib=$(dirname "$(find "$CONDA_PREFIX" -name 'libcudart.so' | head -1)")
echo "libcudart dir: $lib"
if [ ! -e "$CONDA_PREFIX/lib64" ]; then ln -s "$lib" "$CONDA_PREFIX/lib64"; fi
ls -la "$CONDA_PREFIX/lib64" | head -2
