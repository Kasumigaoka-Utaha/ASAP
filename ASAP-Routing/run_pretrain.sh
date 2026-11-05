#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/configs/pretrain_tsp.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG_PATH="$1"
  shift
fi
exec python "${SCRIPT_DIR}/run_config.py" pretrain_policy --config "${CONFIG_PATH}" "$@"
