#!/usr/bin/env bash
set -euo pipefail
bash "$(dirname -- "${BASH_SOURCE[0]}")/launch_seeds.sh" climatology_baseline "$@"
