"""End-to-end sklearn training + MLflow logging."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from workbench.config import TrainConfig
from workbench.datasets import DatasetBundle, load_dataset
from workbench.evaluate import (
    classification_report_text,
    estimator_params,
    evaluate_classification,
    evaluate_regression,
    predict_proba_if_available,
)
from workbench.models import build_model, list_models, unwrap_estimator
from workbench.plots import save_confusion_matrix, save_feature_importance, save_residuals
from workbench.tracking import configure_tracking, log_metrics, log_params


def run_training(cfg: TrainConfig) -> mlflow.entities.Run:
    dataset = load_dataset(cfg.dataset)
    configure_tracking(cfg.tracking_uri, cfg.experiment)
    run_name = cfg.run_name or f"{dataset.name}-{cfg.model}"
    with mlflow.start_run(run_name=run_name) as run:
        _fit_and_log(cfg, dataset)
        return run


def run_comparison(
    datasets: list[str],
    models: list[str] | None,
    *,
    experiment: str,
    tracking_uri: str | None,
    test_size: float,
    random_state: int,
    register_model: bool,
) -> list[str]:
    """Train several dataset/model pairs as nested runs under one parent."""
    configure_tracking(tracking_uri, experiment)
    run_ids: list[str] = []
    parent_name = "compare-" + "-".join(datasets)
    with mlflow.start_run(run_name=parent_name):
        mlflow.set_tags({"workbench.mode": "compare", "datasets": ",".join(datasets)})
        for dataset_name in datasets:
            bundle = load_dataset(dataset_name)
            chosen = models or list_models(bundle.task)
            for model_name in chosen:
                if model_name not in list_models(bundle.task):
                    continue
                cfg = TrainConfig(
                    experiment=experiment,
                    dataset=dataset_name,
                    model=model_name,
                    test_size=test_size,
                    random_state=random_state,
                    tracking_uri=tracking_uri or "",
                    register_model=register_model,
                    run_name=f"{dataset_name}-{model_name}",
                )
                with mlflow.start_run(run_name=cfg.run_name, nested=True) as nested:
                    _fit_and_log(cfg, bundle)
                    run_ids.append(nested.info.run_id)
    return run_ids


def _fit_and_log(cfg: TrainConfig, dataset: DatasetBundle) -> None:
    model = build_model(cfg.model, dataset.task, cfg.params)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=dataset.y if dataset.task == "classification" else None,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    if dataset.task == "classification":
        y_proba = predict_proba_if_available(model, X_test)
        test_metrics = evaluate_classification(y_test, y_pred, y_proba)
        train_metrics = evaluate_classification(y_train, y_train_pred)
    else:
        test_metrics = evaluate_regression(y_test, y_pred)
        train_metrics = evaluate_regression(y_train, y_train_pred)

    uri = cfg.tracking_uri or str(mlflow.get_tracking_uri())
    log_params(
        {
            "dataset": dataset.name,
            "task": dataset.task,
            "model": cfg.model,
            "estimator": type(unwrap_estimator(model)).__name__,
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "n_features": dataset.X.shape[1],
            "n_samples": dataset.X.shape[0],
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
            "tracking_uri": uri,
            **{f"model__{key}": value for key, value in estimator_params(model).items()},
        }
    )
    log_metrics(train_metrics, prefix="train_")
    log_metrics(test_metrics, prefix="test_")
    mlflow.set_tags(
        {
            "dataset": dataset.name,
            "task": dataset.task,
            "model": cfg.model,
            "workbench.version": "0.1.0",
        }
    )
    _log_artifacts(dataset, model, y_test, y_pred, test_metrics)
    _log_sklearn_model(model, X_train, cfg)

    run = mlflow.active_run()
    run_id = run.info.run_id if run else "unknown"
    print(
        f"Run {run_id}  experiment={cfg.experiment}  "
        f"dataset={dataset.name}  model={cfg.model}"
    )
    for key, value in test_metrics.items():
        print(f"  test_{key}={value:.4f}")


def _log_artifacts(
    dataset: DatasetBundle,
    model: object,
    y_test: object,
    y_pred: object,
    test_metrics: dict[str, float],
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(root / "metrics.json"), artifact_path="reports")

        if dataset.task == "classification":
            report = classification_report_text(y_test, y_pred, dataset.target_names)
            (root / "classification_report.txt").write_text(report, encoding="utf-8")
            mlflow.log_artifact(str(root / "classification_report.txt"), artifact_path="reports")
            save_confusion_matrix(
                y_test,
                y_pred,
                root / "confusion_matrix.png",
                labels=dataset.target_names,
            )
            mlflow.log_artifact(str(root / "confusion_matrix.png"), artifact_path="plots")
        else:
            save_residuals(y_test, y_pred, root / "residuals.png")
            mlflow.log_artifact(str(root / "residuals.png"), artifact_path="plots")

        importance = save_feature_importance(model, dataset.features, root / "feature_importance.png")
        if importance is not None:
            mlflow.log_artifact(str(importance), artifact_path="plots")


def _log_sklearn_model(model: object, X_train: object, cfg: TrainConfig) -> None:
    example = X_train.head(min(5, len(X_train)))
    signature = infer_signature(example, model.predict(example))
    registered = cfg.registered_model_name
    common = {
        "sk_model": model,
        "signature": signature,
        "input_example": example,
        "registered_model_name": registered,
    }
    try:
        mlflow.sklearn.log_model(name="model", **common)
    except TypeError:
        mlflow.sklearn.log_model(artifact_path="model", **common)
