#!/usr/bin/env bash
# Environment for running FoundationPose (sourced by the run scripts).
source "${CONDA_HOME:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate foundationpose
export CUDA_HOME="$CONDA_PREFIX" MAX_JOBS=1
# cuDNN dlopens "libnvrtc.so"; conda's CUDA 11.8 ships only the versioned name.
[ -e "$CONDA_PREFIX/lib/libnvrtc.so" ] || ln -s "$CONDA_PREFIX/lib/libnvrtc.so.11.2" "$CONDA_PREFIX/lib/libnvrtc.so"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
