"""Smoke tests against a temporary MLflow store."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from workbench.config import TrainConfig, parse_param_overrides
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


def test_train_logs_metrics_and_model(tmp_path: Path) -> None:
    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    cfg = TrainConfig(
        experiment="workbench-test",
        dataset="iris",
        model="logistic_regression",
        tracking_uri=tracking,
        register_model=True,
        params={"max_iter": 200},
    )
    run = run_training(cfg)
    client = MlflowClient(tracking_uri=tracking)
    logged = client.get_run(run.info.run_id)
    assert logged.data.metrics["test_accuracy"] > 0.7
    assert logged.data.params["dataset"] == "iris"
    artifacts = [item.path for item in client.list_artifacts(run.info.run_id)]
    assert "model" in artifacts or any(path.startswith("model") for path in artifacts)
    models = client.search_registered_models()
    names = {model.name for model in models}
    assert "iris-logistic_regression" in names


def test_compare_creates_nested_runs(tmp_path: Path) -> None:
    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_ids = run_comparison(
        ["iris"],
        ["logistic_regression", "random_forest"],
        experiment="workbench-compare",
        tracking_uri=tracking,
        test_size=0.25,
        random_state=0,
        register_model=False,
    )
    assert len(run_ids) == 2
    mlflow.set_tracking_uri(tracking)
    client = MlflowClient(tracking_uri=tracking)
    exp = client.get_experiment_by_name("workbench-compare")
    assert exp is not None
    runs = client.search_runs(exp.experiment_id)
    names = {run.info.run_name for run in runs}
    assert "compare-iris" in names
    assert "iris-logistic_regression" in names
    assert "iris-random_forest" in names
