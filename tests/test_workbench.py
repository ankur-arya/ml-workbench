"""Smoke tests against the in-process ClearML tracker stand-in."""

from __future__ import annotations

import pytest

from workbench.config import CompareConfig, TrainConfig, parse_param_overrides
from workbench.datasets import list_datasets, load_dataset
from workbench.models import build_model, list_models
from workbench.train import run_comparison, run_training


def test_datasets_and_models_are_consistent() -> None:
    assert set(list_datasets()) == {"iris", "wine", "breast_cancer", "diabetes"}
    iris = load_dataset("iris")
    assert iris.task == "classification"
    assert iris.X.shape[0] == 150
    diabetes = load_dataset("diabetes")
    assert diabetes.task == "regression"
    assert "random_forest" in list_models("classification")
    assert "ridge" in list_models("regression")
    with pytest.raises(ValueError):
        load_dataset("not-a-dataset")


def test_param_overrides() -> None:
    parsed = parse_param_overrides(["n_estimators=50", "max_depth=3", "bootstrap=true"])
    assert parsed == {"n_estimators": 50, "max_depth": 3, "bootstrap": True}


def test_build_model_rejects_wrong_task() -> None:
    with pytest.raises(ValueError, match="not valid"):
        build_model("ridge", "classification")


def test_train_logs_metrics_and_model() -> None:
    cfg = TrainConfig(
        experiment="workbench-test",
        dataset="iris",
        model="logistic_regression",
        register_model=True,
        params={"max_iter": 200},
    )
    result = run_training(cfg)
    assert result.metrics["test_accuracy"] > 0.7
    assert result.params["dataset"] == "iris"
    assert result.model_name == "iris-logistic_regression"
    assert result.model_id
    assert any(path.startswith("model") for path in result.artifacts)
    assert any(path.startswith("plots") for path in result.artifacts)
    assert any(path.startswith("reports") for path in result.artifacts)


def test_compare_creates_sibling_tasks() -> None:
    results = run_comparison(
        ["iris"],
        ["logistic_regression", "random_forest"],
        experiment="workbench-compare",
        test_size=0.25,
        random_state=0,
        register_model=False,
    )
    assert len(results) == 2
    names = {item.task_name for item in results}
    assert names == {"iris-logistic_regression", "iris-random_forest"}
    assert all(item.project == "workbench-compare" for item in results)
    assert all("test_accuracy" in item.metrics for item in results)


def test_compare_skips_incompatible_models() -> None:
    results = run_comparison(
        compare_cfg=CompareConfig(
            experiment="workbench-mixed",
            datasets=["iris", "diabetes"],
            models=["logistic_regression", "ridge"],
            register_model=False,
        )
    )
    names = {item.task_name for item in results}
    assert names == {"iris-logistic_regression", "diabetes-ridge"}
