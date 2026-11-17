#!/bin/sh
# Local CI: code/test gate, then reproduce changed DVC stages and validate packaging.
set -eu
if ! command -v python >/dev/null 2>&1 && [ -x ../.venv/bin/python ]; then
  export PATH="$PWD/../.venv/bin:$PATH"
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m ruff check --no-cache brainai tests scripts
MLFLOW_TRACKING_URI=file:///tmp/brainai-ci-tracking python -m pytest --junitxml=reports/junit.xml
python scripts/test_report.py
if [ "${1:-}" = "--checks-only" ]; then
  exit 0
fi
dvc repro
