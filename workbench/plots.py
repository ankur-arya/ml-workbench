"""Matplotlib artifacts written next to a run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from workbench.models import unwrap_estimator


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    path: Path,
    labels: list[str] | None = None,
) -> Path:
    matrix = confusion_matrix(y_true, y_pred)
    display_labels = labels if labels and len(labels) == matrix.shape[0] else None
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(matrix, display_labels=display_labels).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_residuals(
    y_true: pd.Series,
    y_pred: np.ndarray,
    path: Path,
) -> Path:
    residuals = y_true.to_numpy() - y_pred
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(y_pred, residuals, alpha=0.7, edgecolor="none", color="#3d8bfd")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def save_feature_importance(
    model: object,
    feature_names: list[str],
    path: Path,
    top_n: int = 15,
) -> Path | None:
    estimator = unwrap_estimator(model)
    values: np.ndarray | None = None
    title = "Feature importance"
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
        title = "Mean |coefficient|"
    if values is None or values.size != len(feature_names):
        return None

    order = np.argsort(values)[-top_n:]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(np.asarray(feature_names)[order], values[order], color="#c9a227")
    ax.set_title(title)
    ax.set_xlabel("Weight")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
