"""Train/test metrics for classification and regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from workbench.datasets import TaskType
from workbench.models import unwrap_estimator


def evaluate_classification(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except ValueError:
            pass
    return metrics


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def predict_proba_if_available(model: Any, X: pd.DataFrame) -> np.ndarray | None:
    predict_proba = getattr(model, "predict_proba", None)
    if predict_proba is None:
        return None
    return predict_proba(X)


def classification_report_text(
    y_true: pd.Series,
    y_pred: np.ndarray,
    target_names: list[str] | None,
) -> str:
    labels = sorted(pd.unique(y_true))
    names = None
    if target_names and len(target_names) == len(labels):
        names = target_names
    return classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0)


def estimator_params(model: Any) -> dict[str, Any]:
    inner = unwrap_estimator(model)
    params = inner.get_params(deep=False)
    # MLflow params must be strings / scalars; stringify nested values.
    sanitized: dict[str, Any] = {}
    for key, value in params.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized
