"""End-to-end sklearn training + ClearML logging."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import joblib
from sklearn.model_selection import train_test_split

from workbench import __version__
from workbench.config import CompareConfig, TrainConfig
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
from workbench.tracking import (
    TrainResult,
    TrackingSession,
    open_session,
)

SessionFactory = Callable[..., TrackingSession]


def run_training(
    cfg: TrainConfig,
    *,
    session: TrackingSession | None = None,
    session_factory: SessionFactory | None = None,
    extra_tags: list[str] | None = None,
) -> TrainResult:
    dataset = load_dataset(cfg.dataset)
    owns_session = session is None
    task_name = cfg.run_name or f"{dataset.name}-{cfg.model}"
    tags = _run_tags(cfg, dataset, extra_tags)
    if session is None:
        factory = session_factory or _default_factory
        session = factory(
            project=cfg.project,
            task_name=task_name,
            offline=cfg.offline,
            tags=tags,
            task_type="training",
        )
    try:
        _fit_and_log(cfg, dataset, session)
        return session.handle
    finally:
        if owns_session:
            session.close()


def run_comparison(
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    *,
    experiment: str,
    test_size: float,
    random_state: int,
    register_model: bool,
    register_dataset: bool = False,
    offline: bool = False,
    tags: list[str] | None = None,
    session_factory: SessionFactory | None = None,
    compare_cfg: CompareConfig | None = None,
) -> list[TrainResult]:
    """Train several dataset/model pairs as sibling ClearML tasks in one project.

    A summary task logs a comparison table. Open the project in the ClearML
    WebApp, select the sibling tasks, and click Compare.
    """
    if compare_cfg is not None:
        datasets = compare_cfg.datasets
        models = compare_cfg.models
        experiment = compare_cfg.experiment
        test_size = compare_cfg.test_size
        random_state = compare_cfg.random_state
        register_model = compare_cfg.register_model
        register_dataset = compare_cfg.register_dataset
        offline = compare_cfg.offline
        tags = list(compare_cfg.tags)

    datasets = datasets or ["iris", "wine", "diabetes"]
    factory = session_factory or _default_factory
    group = "compare-" + "-".join(datasets) + "-" + uuid4().hex[:8]
    results: list[TrainResult] = []

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
                register_model=register_model,
                register_dataset=register_dataset,
                offline=offline,
                run_name=f"{dataset_name}-{model_name}",
                tags=list(tags or []),
            )
            results.append(
                run_training(
                    cfg,
                    session_factory=factory,
                    extra_tags=[group, "compare", f"dataset:{dataset_name}", f"model:{model_name}"],
                )
            )

    summary_tags = [group, "compare-summary", *(tags or [])]
    summary = factory(
        project=experiment,
        task_name="compare-" + "-".join(datasets),
        offline=offline,
        tags=summary_tags,
        task_type="controller",
    )
    try:
        rows = [_result_row(item) for item in results]
        summary.set_tags({"workbench.mode": "compare", "datasets": ",".join(datasets), "compare_group": group})
        summary.log_params(
            {
                "datasets": ",".join(datasets),
                "models": ",".join(models or ["(task-default)"]),
                "n_runs": len(results),
                "compare_group": group,
            }
        )
        summary.log_table("comparison", rows)
        for item in results:
            for key, value in item.metrics.items():
                series = f"{item.task_name}/{key}"
                summary.log_metrics({series: value})
        summary.log_text(
            "comparison.json",
            json.dumps(rows, indent=2),
            artifact_path="reports",
        )
    finally:
        summary.close()

    return results


def _fit_and_log(cfg: TrainConfig, dataset: DatasetBundle, session: TrackingSession) -> None:
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

    session.log_params(
        {
            "dataset": dataset.name,
            "task": dataset.task,
            "model": cfg.model,
            "estimator": type(unwrap_estimator(model)).__name__,
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "n_features": int(dataset.X.shape[1]),
            "n_samples": int(dataset.X.shape[0]),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            **{f"model__{key}": value for key, value in estimator_params(model).items()},
        }
    )
    session.log_metrics(train_metrics, prefix="train_")
    session.log_metrics(test_metrics, prefix="test_")
    session.set_tags(
        {
            "dataset": dataset.name,
            "task": dataset.task,
            "model": cfg.model,
            "workbench.version": __version__,
        }
    )
    _log_artifacts(dataset, model, y_test, y_pred, test_metrics, session, cfg)
    _print_summary(session, cfg, dataset, test_metrics)


def _log_artifacts(
    dataset: DatasetBundle,
    model: object,
    y_test: object,
    y_pred: object,
    test_metrics: dict[str, float],
    session: TrackingSession,
    cfg: TrainConfig,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        session.log_artifact(root / "metrics.json", artifact_path="reports")

        preview = dataset.frame.head(25)
        preview.to_csv(root / "dataset_preview.csv", index=False)
        session.log_artifact(root / "dataset_preview.csv", artifact_path="data")

        if cfg.register_dataset:
            dataset.frame.to_csv(root / f"{dataset.name}.csv", index=False)
            session.register_dataset(
                dataset.name,
                root / f"{dataset.name}.csv",
                preview_path=root / "dataset_preview.csv",
            )

        if dataset.task == "classification":
            report = classification_report_text(y_test, y_pred, dataset.target_names)
            session.log_text("classification_report.txt", report, artifact_path="reports")
            save_confusion_matrix(
                y_test,
                y_pred,
                root / "confusion_matrix.png",
                labels=dataset.target_names,
            )
            session.log_artifact(root / "confusion_matrix.png", artifact_path="plots")
        else:
            save_residuals(y_test, y_pred, root / "residuals.png")
            session.log_artifact(root / "residuals.png", artifact_path="plots")

        importance = save_feature_importance(model, dataset.features, root / "feature_importance.png")
        if importance is not None:
            session.log_artifact(importance, artifact_path="plots")

        model_path = root / "model.joblib"
        joblib.dump(model, model_path)
        session.log_artifact(model_path, artifact_path="model")
        if cfg.register_model and cfg.registered_model_name:
            comment = json.dumps({"dataset": dataset.name, "model": cfg.model, "metrics": test_metrics})
            session.register_model(
                model_path,
                cfg.registered_model_name,
                tags=["candidate", dataset.name, cfg.model, dataset.task],
                comment=comment,
            )


def _print_summary(
    session: TrackingSession,
    cfg: TrainConfig,
    dataset: DatasetBundle,
    test_metrics: dict[str, float],
) -> None:
    handle = session.handle
    print(
        f"Task {handle.task_id}  project={cfg.experiment}  "
        f"dataset={dataset.name}  model={cfg.model}"
    )
    if handle.model_id:
        print(f"  registered_model={handle.model_name}  model_id={handle.model_id}")
    if handle.web_url:
        print(f"  webapp={handle.web_url}")
    for key, value in test_metrics.items():
        print(f"  test_{key}={value:.4f}")


def _run_tags(cfg: TrainConfig, dataset: DatasetBundle, extra: list[str] | None) -> list[str]:
    tags = [
        *cfg.tags,
        dataset.name,
        cfg.model,
        dataset.task,
        *(extra or []),
    ]
    seen: list[str] = []
    for tag in tags:
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def _result_row(result: TrainResult) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": result.task_id,
        "task_name": result.task_name,
        "project": result.project,
        "model_id": result.model_id or "",
        "model_name": result.model_name or "",
    }
    row.update(result.metrics)
    row.update({f"param_{key}": value for key, value in result.params.items() if key in {"dataset", "model", "task"}})
    return row


def _default_factory(**kwargs: object) -> TrackingSession:
    return open_session(
        project=str(kwargs["project"]),
        task_name=str(kwargs["task_name"]),
        offline=bool(kwargs.get("offline", False)),
        tags=kwargs.get("tags"),  # type: ignore[arg-type]
        task_type=str(kwargs.get("task_type") or "training"),
    )
