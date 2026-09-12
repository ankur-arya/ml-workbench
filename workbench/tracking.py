"""MLflow tracking helpers (local SQLite by default)."""

from __future__ import annotations

import os
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from workbench.config import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI


def resolve_tracking_uri(explicit: str | None = None) -> str:
    return explicit or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def configure_tracking(
    tracking_uri: str | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
) -> str:
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def log_params(params: dict[str, Any]) -> None:
    mlflow.log_params({key: _stringify(value) for key, value in params.items()})


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    payload = {f"{prefix}{key}": float(value) for key, value in metrics.items()}
    mlflow.log_metrics(payload)


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def latest_run_url(tracking_uri: str, experiment: str) -> str:
    """Best-effort deep link when a local MLflow UI is assumed on :5000."""
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return "http://127.0.0.1:5000"
    return f"http://127.0.0.1:5000/#/experiments/{exp.experiment_id}"
