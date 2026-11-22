#!/usr/bin/env bash
# One command to reproduce everything inside the Docker image:
#   pull the versioned data from DagsHub, re-run every pipeline stage, show the metrics.
set -euo pipefail
git config --global --add safe.directory /app || true

if [[ -n "${DAGSHUB_USER:-}" && -n "${DAGSHUB_TOKEN:-}" ]]; then
  # needed for private repos and for logging to DagsHub MLflow
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user "$DAGSHUB_USER"
  dvc remote modify origin --local password "$DAGSHUB_TOKEN"
  export MLFLOW_TRACKING_USERNAME="$DAGSHUB_USER"
  export MLFLOW_TRACKING_PASSWORD="$DAGSHUB_TOKEN"
fi

dvc pull data/raw.dvc          # raw dataset only; everything else is rebuilt
python -m pytest -q            # leakage / validation unit tests
dvc repro --force              # re-run all stages: validate -> features -> windows -> train
dvc metrics show
