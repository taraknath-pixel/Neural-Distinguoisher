#!/usr/bin/env bash
# Linux equivalent of 3_run_pipeline__ml_eval.bat.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PROJECT_ROOT}/scripts/run_evaluate.sh" "$@"
