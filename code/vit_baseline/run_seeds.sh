#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname -- "${BASH_SOURCE[0]}")/../climatology_baseline/launch_seeds.sh" vit_baseline "$@"
