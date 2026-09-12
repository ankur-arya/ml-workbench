"""Leaderboard ranking and recommended-winner logic."""

from __future__ import annotations

import pytest

from workbench.config import Constraint
from workbench.ranking import compare_experiment, rank_runs
from workbench.store import Store


def _run(store: Store, experiment_id: str, model: str, accuracy: float, f1: float = 0.9, dataset: str = "iris"):
    run = store.create_run(
        experiment_id=experiment_id,
        name=f"{dataset}-{model}",
        model_name=model,
        dataset_id=dataset,
        task="classification",
        status="succeeded",
    )
    store.set_metrics(run["id"], {"test_accuracy": accuracy, "test_f1_macro": f1})
    store.set_params(run["id"], {"model": model, "dataset": dataset, "model__n_estimators": 10})
    return store.get_run(run["id"])


def test_rank_runs_orders_by_primary_metric_desc(store: Store) -> None:
    experiment = store.create_experiment(name="rank", primary_metric="test_accuracy", maximize=True)
    low = _run(store, experiment["id"], "logistic_regression", 0.80)
    high = _run(store, experiment["id"], "random_forest", 0.96)
    mid = _run(store, experiment["id"], "gradient_boosting", 0.90)
    ranked = rank_runs([low, high, mid], primary_metric="test_accuracy", maximize=True)
    assert [row["model_name"] for row in ranked] == [
        "random_forest",
        "gradient_boosting",
        "logistic_regression",
    ]
    assert ranked[0]["is_recommended"] is True
    assert ranked[0]["rank"] == 1
    assert ranked[1]["delta_vs_best"] == pytest.approx(-0.06)


def test_constraints_drop_ineligible_winner(store: Store) -> None:
    experiment = store.create_experiment(name="constrained", primary_metric="test_accuracy", maximize=True)
    fast = _run(store, experiment["id"], "logistic_regression", 0.99, f1=0.50)
    solid = _run(store, experiment["id"], "random_forest", 0.93, f1=0.92)
    ranked = rank_runs(
        [fast, solid],
        primary_metric="test_accuracy",
        maximize=True,
        constraints=[Constraint(metric="test_f1_macro", min=0.8)],
    )
    recommended = next(row for row in ranked if row["is_recommended"])
    assert recommended["model_name"] == "random_forest"
    dropped = next(row for row in ranked if row["model_name"] == "logistic_regression")
    assert dropped["constraint_failed"] is True


def test_compare_experiment_recommends_winner(store: Store) -> None:
    experiment = store.create_experiment(
        name="iris-bakeoff",
        primary_metric="test_accuracy",
        maximize=True,
    )
    _run(store, experiment["id"], "logistic_regression", 0.86)
    _run(store, experiment["id"], "random_forest", 0.97)
    scorecard = compare_experiment(store.get_experiment(experiment["id"]), store.list_runs(experiment["id"]))
    assert scorecard["recommended"]["model_name"] == "random_forest"
    assert scorecard["primary_metric"] == "test_accuracy"
    assert scorecard["leaderboard"][0]["is_recommended"] is True


def test_regression_defaults_to_r2(store: Store) -> None:
    experiment = store.create_experiment(name="diabetes")
    run = store.create_run(
        experiment_id=experiment["id"],
        name="diabetes-ridge",
        model_name="ridge",
        dataset_id="diabetes",
        task="regression",
        status="succeeded",
    )
    store.set_metrics(run["id"], {"test_r2": 0.44, "test_rmse": 55.0})
    scorecard = compare_experiment(store.get_experiment(experiment["id"]), store.list_runs(experiment["id"]))
    assert scorecard["primary_metric"] == "test_r2"
    assert scorecard["recommended"]["model_name"] == "ridge"
