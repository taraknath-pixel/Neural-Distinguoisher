#!/usr/bin/env bash
# Run binary and multiclass evaluation from an existing extracted_features.npz file.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON:-python3}" -m crypto_identifier.pipeline evaluate "$@"
