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


def list_models(task: TaskType | None = None) -> list[str]:
    if task == "classification":
        return list(CLASSIFICATION_MODELS)
    if task == "regression":
        return list(REGRESSION_MODELS)
    return sorted(set(CLASSIFICATION_MODELS) | set(REGRESSION_MODELS))


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

    merged = {**defaults, **params}
    estimator = catalog[name](**merged)
    return estimator


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
