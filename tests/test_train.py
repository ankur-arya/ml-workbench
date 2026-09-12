"""End-to-end training against the local SQLite store."""

from __future__ import annotations

from workbench.config import CompareConfig, ModelSpec, TrainConfig
from workbench.store import Store
from workbench.train import experiment_scorecard, run_comparison, run_training


def test_train_logs_metrics_and_model(store: Store) -> None:
    cfg = TrainConfig(
        experiment="workbench-test",
        dataset="iris",
        model="logistic_regression",
        register_model=True,
        params={"max_iter": 200},
    )
    run = run_training(cfg, store=store)
    assert run["status"] == "succeeded"
    assert run["metrics"]["test_accuracy"] > 0.7
    assert run["params"]["dataset"] == "iris"
    names = {item["name"] for item in run["artifacts"]}
    assert "model.joblib" in names
    assert "confusion_matrix.json" in names
    versions = store.versions_for_run(run["id"])
    assert versions
    assert versions[0]["stage"] == "candidate"


def test_compare_ranks_multiple_models(store: Store) -> None:
    cfg = CompareConfig(
        experiment="iris-compare",
        datasets=["iris"],
        models=[
            ModelSpec(name="logistic_regression", params={"max_iter": 200}),
            ModelSpec(name="random_forest", params={"n_estimators": 20, "max_depth": 3}),
        ],
        register_model=False,
        test_size=0.25,
        random_state=0,
        primary_metric="test_accuracy",
        maximize=True,
    )
    experiment = run_comparison(cfg, store=store)
    assert experiment["status"] == "completed"
    runs = store.list_runs(experiment["id"])
    assert len(runs) == 2
    scorecard = experiment_scorecard(store, experiment["id"])
    assert scorecard["recommended"] is not None
    assert scorecard["recommended"]["rank"] == 1
    assert {run["model_name"] for run in runs} == {"logistic_regression", "random_forest"}
