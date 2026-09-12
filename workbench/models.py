"""Build sklearn estimators (optionally inside a scaling pipeline)."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from workbench.datasets import TaskType

CLASSIFICATION_MODELS = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "svc",
)
REGRESSION_MODELS = (
    "linear_regression",
    "ridge",
    "lasso",
    "random_forest",
    "gradient_boosting",
    "svr",
)

# Linear / kernel models benefit from feature scaling; tree models do not.
_SCALE_MODELS = {
    "logistic_regression",
    "svc",
    "linear_regression",
    "ridge",
    "lasso",
    "svr",
}

MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "blurb": "Fast linear baseline. Strong when classes are roughly linearly separable.",
        "tasks": ["classification"],
        "params": [
            {"key": "C", "label": "C", "type": "float", "default": 1.0, "min": 0.01, "max": 100},
            {"key": "max_iter", "label": "Max iter", "type": "int", "default": 500, "min": 100, "max": 4000},
        ],
    },
    "random_forest": {
        "label": "Random Forest",
        "blurb": "Bagged trees. Robust default for tabular classification and regression.",
        "tasks": ["classification", "regression"],
        "params": [
            {"key": "n_estimators", "label": "Trees", "type": "int", "default": 100, "min": 10, "max": 500},
            {"key": "max_depth", "label": "Max depth", "type": "int", "default": None, "min": 1, "max": 40},
        ],
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "blurb": "Stage-wise trees. Usually the accuracy pick on small sklearn tables.",
        "tasks": ["classification", "regression"],
        "params": [
            {"key": "n_estimators", "label": "Stages", "type": "int", "default": 100, "min": 20, "max": 400},
            {"key": "learning_rate", "label": "Learn rate", "type": "float", "default": 0.1, "min": 0.01, "max": 1},
            {"key": "max_depth", "label": "Max depth", "type": "int", "default": 3, "min": 1, "max": 8},
        ],
    },
    "svc": {
        "label": "Support Vector (C)",
        "blurb": "RBF kernel classifier. Slower, sharp decision boundaries.",
        "tasks": ["classification"],
        "params": [
            {"key": "C", "label": "C", "type": "float", "default": 1.0, "min": 0.01, "max": 100},
        ],
    },
    "linear_regression": {
        "label": "Linear Regression",
        "blurb": "Ordinary least squares. The honest regression baseline.",
        "tasks": ["regression"],
        "params": [],
    },
    "ridge": {
        "label": "Ridge",
        "blurb": "L2-regularized linear regression. Stable when features correlate.",
        "tasks": ["regression"],
        "params": [
            {"key": "alpha", "label": "Alpha", "type": "float", "default": 1.0, "min": 0.0001, "max": 100},
        ],
    },
    "lasso": {
        "label": "Lasso",
        "blurb": "L1-regularized linear regression. Promotes sparse coefficients.",
        "tasks": ["regression"],
        "params": [
            {"key": "alpha", "label": "Alpha", "type": "float", "default": 0.1, "min": 0.0001, "max": 10},
        ],
    },
    "svr": {
        "label": "Support Vector (R)",
        "blurb": "RBF kernel regressor for non-linear residual structure.",
        "tasks": ["regression"],
        "params": [
            {"key": "C", "label": "C", "type": "float", "default": 1.0, "min": 0.01, "max": 100},
        ],
    },
}


def list_models(task: TaskType | None = None) -> list[str]:
    if task == "classification":
        return list(CLASSIFICATION_MODELS)
    if task == "regression":
        return list(REGRESSION_MODELS)
    return sorted(set(CLASSIFICATION_MODELS) | set(REGRESSION_MODELS))


def catalog_for_ui() -> list[dict[str, Any]]:
    items = []
    for key in list_models():
        meta = MODEL_CATALOG[key]
        items.append({"key": key, **meta})
    return items


def _estimator_for(name: str, task: TaskType, params: dict[str, Any]) -> BaseEstimator:
    if task == "classification":
        catalog: dict[str, type[BaseEstimator]] = {
            "logistic_regression": LogisticRegression,
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "svc": SVC,
        }
    else:
        catalog = {
            "linear_regression": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor,
            "svr": SVR,
        }

    if name not in catalog:
        available = ", ".join(list_models(task))
        raise ValueError(
            f"Model {name!r} is not valid for task {task!r}. Choose one of: {available}"
        )

    defaults: dict[str, Any] = {"random_state": 42}
    if name == "logistic_regression":
        defaults.update({"max_iter": 500, "solver": "lbfgs"})
    elif name == "svc":
        defaults.update({"probability": True, "kernel": "rbf"})
    elif name in {"random_forest", "gradient_boosting"}:
        defaults.update({"n_estimators": 100})
    elif name == "ridge":
        defaults.update({"alpha": 1.0})
    elif name == "lasso":
        defaults.update({"alpha": 0.1, "max_iter": 5000})

    if name in {"linear_regression"}:
        defaults.pop("random_state", None)

    merged = {**defaults, **{k: v for k, v in params.items() if v is not None}}
    return catalog[name](**merged)


def build_model(
    name: str,
    task: TaskType,
    params: dict[str, Any] | None = None,
    *,
    scale: bool | None = None,
) -> Pipeline:
    """Return a sklearn Pipeline ready to fit.

    Tree ensembles skip StandardScaler unless ``scale`` is forced on.
    """
    key = name.strip().lower()
    estimator = _estimator_for(key, task, params or {})
    use_scaler = _SCALE_MODELS.__contains__(key) if scale is None else scale
    steps: list[tuple[str, BaseEstimator]] = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def unwrap_estimator(model: BaseEstimator) -> BaseEstimator:
    """Return the inner estimator if ``model`` is a Pipeline."""
    if isinstance(model, Pipeline):
        return model.named_steps["model"]
    return model
