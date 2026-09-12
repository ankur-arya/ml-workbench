"""Fit sklearn models and record them in the local workbench store."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from typing import Any

import joblib
from sklearn.model_selection import train_test_split

from workbench.config import CompareConfig, ModelSpec, TrainConfig, default_primary_for_task
from workbench.datasets import DatasetBundle, load_dataset, load_tabular
from workbench.evaluate import (
    classification_report_text,
    confusion_payload,
    estimator_params,
    evaluate_classification,
    evaluate_regression,
    predict_proba_if_available,
    residual_payload,
)
from workbench.models import build_model, list_models, unwrap_estimator
from workbench.paths import get_store
from workbench.plots import save_confusion_matrix, save_feature_importance, save_residuals
from workbench.ranking import compare_experiment
from workbench.registry import register_run
from workbench.store import Store, utcnow


def resolve_bundle(store: Store, dataset_ref: str) -> tuple[DatasetBundle, dict[str, Any]]:
    record = store.get_dataset(dataset_ref)
    if record["source"] == "builtin":
        bundle = load_dataset(record["slug"])
    else:
        if not record.get("path"):
            raise ValueError(f"Dataset {dataset_ref} has no file path")
        bundle = load_tabular(
            record["path"],
            name=record["slug"],
            target=record["target"],
            task=record["task"],
            description=record.get("description") or "",
        )
    return bundle, record


def run_training(cfg: TrainConfig, store: Store | None = None) -> dict[str, Any]:
    store = store or get_store()
    bundle, dataset = resolve_bundle(store, cfg.dataset)
    primary, maximize = _primary_for(cfg.primary_metric, cfg.maximize, bundle.task)
    experiment = store.create_experiment(
        name=cfg.experiment,
        description=cfg.description,
        primary_metric=primary,
        maximize=maximize,
        spec={
            "datasets": [cfg.dataset],
            "models": [{"name": cfg.model, "params": cfg.params}],
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "tags": cfg.tags,
        },
        status="running",
    )
    run = _execute_run(
        store,
        experiment_id=experiment["id"],
        dataset=dataset,
        bundle=bundle,
        model_name=cfg.model,
        params=cfg.params,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        run_name=cfg.run_name,
        register_model=cfg.register_model,
        tags=cfg.tags,
    )
    status = "completed" if run["status"] == "succeeded" else "failed"
    store.update_experiment(experiment["id"], status=status, error=run.get("error"))
    return store.get_run(run["id"])


def run_comparison(cfg: CompareConfig, store: Store | None = None) -> dict[str, Any]:
    store = store or get_store()
    datasets = list(cfg.datasets)
    specs = cfg.models
    first_task = store.get_dataset(datasets[0])["task"]
    primary, maximize = _primary_for(cfg.primary_metric, cfg.maximize, first_task)
    experiment = store.create_experiment(
        name=cfg.experiment,
        description=cfg.description,
        primary_metric=primary,
        maximize=maximize,
        constraints=[asdict(item) for item in cfg.constraints],
        spec={
            "datasets": datasets,
            "models": [asdict(item) for item in (specs or [])],
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "tags": cfg.tags,
        },
        status="running",
    )
    _run_experiment_jobs(store, experiment["id"], cfg)
    return store.get_experiment(experiment["id"])


def start_experiment_from_spec(store: Store, spec: dict[str, Any], *, background: bool = False) -> dict[str, Any]:
    cfg = _compare_from_spec(spec)
    first_task = store.get_dataset(cfg.datasets[0])["task"]
    primary, maximize = _primary_for(cfg.primary_metric, cfg.maximize, first_task)
    experiment = store.create_experiment(
        name=cfg.experiment,
        description=cfg.description,
        primary_metric=primary,
        maximize=maximize,
        constraints=[asdict(item) for item in cfg.constraints],
        spec=spec,
        status="running",
    )
    if background:
        _spawn(store, experiment["id"], cfg)
    else:
        _run_experiment_jobs(store, experiment["id"], cfg)
    return store.get_experiment(experiment["id"])


def _compare_from_spec(spec: dict[str, Any]) -> CompareConfig:
    from workbench.config import compare_config_from_mapping

    payload = dict(spec)
    if "name" in payload and "experiment" not in payload:
        payload["experiment"] = payload["name"]
    return compare_config_from_mapping(payload)


def _spawn(store: Store, experiment_id: str, cfg: CompareConfig) -> None:
    import threading

    thread = threading.Thread(
        target=_run_experiment_jobs,
        args=(store, experiment_id, cfg),
        name=f"workbench-{experiment_id}",
        daemon=True,
    )
    thread.start()


def _run_experiment_jobs(store: Store, experiment_id: str, cfg: CompareConfig) -> None:
    failed = False
    try:
        for dataset_ref in cfg.datasets:
            bundle, dataset = resolve_bundle(store, dataset_ref)
            specs = cfg.models or [ModelSpec(name=name) for name in list_models(bundle.task)]
            for spec in specs:
                if spec.name not in list_models(bundle.task):
                    continue
                run = _execute_run(
                    store,
                    experiment_id=experiment_id,
                    dataset=dataset,
                    bundle=bundle,
                    model_name=spec.name,
                    params=spec.params,
                    test_size=cfg.test_size,
                    random_state=cfg.random_state,
                    run_name=f"{dataset['slug']}-{spec.name}",
                    register_model=cfg.register_model,
                    tags=cfg.tags,
                )
                if run["status"] != "succeeded":
                    failed = True
        store.update_experiment(experiment_id, status="failed" if failed else "completed")
    except Exception as exc:  # noqa: BLE001 - surface into the experiment row
        store.update_experiment(experiment_id, status="failed", error=str(exc))
        raise


def _execute_run(
    store: Store,
    *,
    experiment_id: str,
    dataset: dict[str, Any],
    bundle: DatasetBundle,
    model_name: str,
    params: dict[str, Any],
    test_size: float,
    random_state: int,
    run_name: str | None,
    register_model: bool,
    tags: list[str],
) -> dict[str, Any]:
    run = store.create_run(
        experiment_id=experiment_id,
        name=run_name or f"{dataset['slug']}-{model_name}",
        model_name=model_name,
        dataset_id=dataset["id"],
        task=bundle.task,
        status="running",
    )
    store.update_run(run["id"], started_at=utcnow())
    try:
        _fit_and_record(
            store,
            run_id=run["id"],
            bundle=bundle,
            dataset=dataset,
            model_name=model_name,
            params=params,
            test_size=test_size,
            random_state=random_state,
            tags=tags,
        )
        finished = store.update_run(run["id"], status="succeeded", finished_at=utcnow())
        if register_model:
            register_run(store, finished)
        _print_run(finished)
        return store.get_run(run["id"])
    except Exception as exc:  # noqa: BLE001
        store.update_run(
            run["id"],
            status="failed",
            finished_at=utcnow(),
            error=f"{exc}\n{traceback.format_exc()}",
        )
        return store.get_run(run["id"])


def _fit_and_record(
    store: Store,
    *,
    run_id: str,
    bundle: DatasetBundle,
    dataset: dict[str, Any],
    model_name: str,
    params: dict[str, Any],
    test_size: float,
    random_state: int,
    tags: list[str],
) -> None:
    model = build_model(model_name, bundle.task, params)
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=test_size,
        random_state=random_state,
        stratify=bundle.y if bundle.task == "classification" else None,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    if bundle.task == "classification":
        y_proba = predict_proba_if_available(model, X_test)
        test_metrics = evaluate_classification(y_test, y_pred, y_proba)
        train_metrics = evaluate_classification(y_train, y_train_pred)
    else:
        test_metrics = evaluate_regression(y_test, y_pred)
        train_metrics = evaluate_regression(y_train, y_train_pred)

    store.set_params(
        run_id,
        {
            "dataset": dataset["slug"],
            "task": bundle.task,
            "model": model_name,
            "estimator": type(unwrap_estimator(model)).__name__,
            "test_size": test_size,
            "random_state": random_state,
            "n_features": int(bundle.X.shape[1]),
            "n_samples": int(bundle.X.shape[0]),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "tags": tags,
            **{f"model__{key}": value for key, value in estimator_params(model).items()},
        },
    )
    prefixed = {f"train_{key}": value for key, value in train_metrics.items()}
    prefixed.update({f"test_{key}": value for key, value in test_metrics.items()})
    store.set_metrics(run_id, prefixed)
    _write_artifacts(store, run_id, bundle, model, y_test, y_pred, test_metrics)


def _write_artifacts(
    store: Store,
    run_id: str,
    bundle: DatasetBundle,
    model: object,
    y_test: object,
    y_pred: object,
    test_metrics: dict[str, float],
) -> None:
    root = store.artifact_dir(run_id)
    metrics_path = root / "metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    store.add_artifact(run_id, name="metrics.json", kind="report", path=str(metrics_path), mime="application/json", label="Metrics")

    model_path = root / "model.joblib"
    joblib.dump(model, model_path)
    store.add_artifact(run_id, name="model.joblib", kind="model", path=str(model_path), mime="application/octet-stream", label="sklearn model")

    if bundle.task == "classification":
        report = classification_report_text(y_test, y_pred, bundle.target_names)
        report_path = root / "classification_report.txt"
        report_path.write_text(report, encoding="utf-8")
        store.add_artifact(run_id, name="classification_report.txt", kind="report", path=str(report_path), mime="text/plain", label="Classification report")

        matrix = confusion_payload(y_test, y_pred, bundle.target_names)
        matrix_json = root / "confusion_matrix.json"
        matrix_json.write_text(json.dumps(matrix), encoding="utf-8")
        store.add_artifact(run_id, name="confusion_matrix.json", kind="json", path=str(matrix_json), mime="application/json", label="Confusion matrix")

        png = root / "confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, png, labels=bundle.target_names)
        store.add_artifact(run_id, name="confusion_matrix.png", kind="plot", path=str(png), mime="image/png", label="Confusion matrix")
    else:
        residuals = residual_payload(y_test, y_pred)
        residual_json = root / "residuals.json"
        residual_json.write_text(json.dumps(residuals), encoding="utf-8")
        store.add_artifact(run_id, name="residuals.json", kind="json", path=str(residual_json), mime="application/json", label="Residuals")
        png = root / "residuals.png"
        save_residuals(y_test, y_pred, png)
        store.add_artifact(run_id, name="residuals.png", kind="plot", path=str(png), mime="image/png", label="Residuals")

    importance = save_feature_importance(model, bundle.features, root / "feature_importance.png")
    if importance is not None:
        store.add_artifact(run_id, name="feature_importance.png", kind="plot", path=str(importance), mime="image/png", label="Feature importance")


def _primary_for(explicit: str | None, maximize: bool | None, task: str) -> tuple[str, bool]:
    default_metric, default_max = default_primary_for_task(task)
    return explicit or default_metric, default_max if maximize is None else bool(maximize)


def _print_run(run: dict[str, Any]) -> None:
    print(
        f"Run {run['id']}  experiment={run['experiment_id']}  "
        f"dataset={(run.get('dataset') or {}).get('slug')}  model={run['model_name']}"
    )
    for key, value in sorted((run.get("metrics") or {}).items()):
        if key.startswith("test_"):
            print(f"  {key}={value:.4f}")


def experiment_scorecard(store: Store, experiment_id: str, dataset: str | None = None) -> dict[str, Any]:
    experiment = store.get_experiment(experiment_id)
    runs = store.list_runs(experiment_id)
    return compare_experiment(experiment, runs, dataset=dataset)


def next_pending_label(store: Store, experiment_id: str) -> str | None:
    runs = store.list_runs(experiment_id)
    current = next((run for run in runs if run["status"] == "running"), None)
    if current:
        return current["name"]
    pending = next((run for run in runs if run["status"] == "pending"), None)
    return pending["name"] if pending else None
