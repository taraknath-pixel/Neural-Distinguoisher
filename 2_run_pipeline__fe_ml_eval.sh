#!/usr/bin/env bash
# Linux equivalent of 2_run_pipeline__fe_ml_eval.bat.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PROJECT_ROOT}/scripts/run_features_and_evaluate.sh" "$@"
