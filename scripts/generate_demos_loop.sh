#!/usr/bin/env bash
# Keep generating grasp demos in short-lived processes (25 episodes each, then exit so
# the memory is returned; a long-running generator died of MemoryError). Usage:
#   bash scripts/generate_demos_loop.sh <first seed> <number of batches>
first=${1:-100}; batches=${2:-20}
for ((i=0; i<batches; i++)); do
  seed=$((first + i))
  python -m scripts.generate_grasp_demos --count 25 --seed "$seed" --out artifacts/demos >> "artifacts/demos_loop_${first}.log" 2>&1
done
