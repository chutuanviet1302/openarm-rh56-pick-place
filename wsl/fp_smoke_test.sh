#!/usr/bin/env bash
# Run FoundationPose's own mustard0 demo on the GTX 1650 and record time + GPU memory.
# Settings reduced for 4 GB: est_refine_iter 3 (register), track_refine_iter 2.
# If register runs out of memory, lower the rotation hypotheses in estimater.py
# (make_rotation_grid(min_n_views=40, inplane_step=60) -> min_n_views=20, inplane_step=90).
set -euo pipefail
FP_HOME="${FP_HOME:-$HOME/foundationpose}"
source "${CONDA_HOME:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate foundationpose
cd "$FP_HOME/FoundationPose"
LOG="$FP_HOME/smoke_test.log"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > "$FP_HOME/gpu_mem.log" &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT
/usr/bin/time -v python run_demo.py --mesh_file demo_data/mustard0/mesh/textured_simple.obj \
  --test_scene_dir demo_data/mustard0 --est_refine_iter 3 --track_refine_iter 2 --debug 1 \
  --debug_dir "$FP_HOME/debug_mustard0" 2>&1 | tee "$LOG"
echo "peak GPU memory (MiB): $(sort -n "$FP_HOME/gpu_mem.log" | tail -1)"
echo "frames tracked: $(ls "$FP_HOME/debug_mustard0/ob_in_cam" 2>/dev/null | wc -l)"
grep -E "Elapsed \(wall|Maximum resident" "$LOG" || true
