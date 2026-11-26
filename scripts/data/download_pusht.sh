#!/usr/bin/env bash
# Run in the foreground or with nohup; the Python script reports every 15 seconds.
set -euo pipefail
source /etc/network_turbo
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_PYTHON="${DATA_PYTHON:-python3}"
exec "$DATA_PYTHON" -u "$SCRIPT_DIR/download_pusht.py" "$@"
