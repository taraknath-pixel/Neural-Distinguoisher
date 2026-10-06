#!/usr/bin/env bash
# Generate the dataset, extract NIST features, then run binary and multiclass evaluation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON:-python3}" -m crypto_identifier.pipeline all "$@"
