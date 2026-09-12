"""Opinionated first-run experiment: three iris classifiers."""

from __future__ import annotations

from workbench.config import CompareConfig, ModelSpec
from workbench.store import Store
from workbench.train import run_comparison, start_experiment_from_spec

DEMO_SPEC = {
    "name": "Iris bakeoff",
    "experiment": "Iris bakeoff",
    "description": "Three sklearn classifiers on iris. The recommended winner is the one to promote.",
    "datasets": ["iris"],
    "models": [
        {"name": "logistic_regression", "params": {"max_iter": 400}},
        {"name": "random_forest", "params": {"n_estimators": 80, "max_depth": 4}},
        {"name": "gradient_boosting", "params": {"n_estimators": 80, "max_depth": 2}},
    ],
    "primary_metric": "test_accuracy",
    "maximize": True,
    "test_size": 0.25,
    "random_state": 42,
    "register_model": True,
    "tags": ["demo"],
}


def demo_config() -> CompareConfig:
    return CompareConfig(
        experiment="Iris bakeoff",
        description=DEMO_SPEC["description"],
        datasets=["iris"],
        models=[ModelSpec(name=item["name"], params=item.get("params") or {}) for item in DEMO_SPEC["models"]],
        test_size=0.25,
        random_state=42,
        register_model=True,
        tags=["demo"],
        primary_metric="test_accuracy",
        maximize=True,
    )


def run_demo(store: Store, *, background: bool = False) -> dict:
    if background:
        return start_experiment_from_spec(store, DEMO_SPEC, background=True)
    return run_comparison(demo_config(), store=store)
