#!/usr/bin/env bash
# Extract features from an existing dataset, then run binary and multiclass evaluation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON:-python3}" -m crypto_identifier.pipeline features-evaluate "$@"
