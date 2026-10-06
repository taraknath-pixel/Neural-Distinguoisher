#!/usr/bin/env bash
# Linux equivalent of 1_run_pipeline_ds_fe_ml_eval.bat.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PROJECT_ROOT}/scripts/run_all.sh" "$@"
