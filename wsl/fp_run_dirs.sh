#!/usr/bin/env bash
# Run wsl/fp_run.py on frame directories (WSL paths).
set -eo pipefail
source "$(dirname "$0")/fp_env.sh"
python "$(dirname "$0")/fp_run.py" "$@"
