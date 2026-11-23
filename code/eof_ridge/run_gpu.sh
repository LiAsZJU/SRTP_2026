#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname -- "${BASH_SOURCE[0]}")/../climatology_baseline/launch_model.sh" eof_ridge "$@"
