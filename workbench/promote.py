"""Promote a ClearML output model toward production (tag + publish).

ClearML's recommended pattern is: pick the winning OutputModel in the model
catalog, add a ``production`` / ``champion`` tag, and **Publish** it so the
artifact becomes immutable and usable as a registry entry (and as a trigger
for ClearML Serving).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from workbench.tracking import flatten_scalars

LOWER_IS_BETTER = {
    "mae",
    "mse",
    "rmse",
    "test_mae",
    "test_mse",
    "test_rmse",
    "train_mae",
    "train_mse",
    "train_rmse",
    "loss",
}


@dataclass
class PromoteResult:
    model_id: str
    model_name: str | None
    task_id: str | None
    tags: list[str] = field(default_factory=list)
    published: bool = False
    metric: str | None = None
    metric_value: float | None = None


class ModelRegistry(Protocol):
    def get_model(self, model_id: str) -> Any: ...
    def models_for_task(self, task_id: str) -> list[Any]: ...
    def list_tasks(self, project: str, tags: list[str] | None = None) -> list[Any]: ...
    def task_metrics(self, task: Any) -> dict[str, float]: ...
    def task_output_models(self, task: Any) -> list[Any]: ...
    def apply_tags(self, model: Any, tags: list[str]) -> list[str]: ...
    def publish(self, model: Any) -> None: ...


class ClearMLRegistry:
    """Thin wrapper around ``clearml.Model`` / ``clearml.Task``."""

    def get_model(self, model_id: str) -> Any:
        from clearml import Model

        return Model(model_id=model_id)

    def models_for_task(self, task_id: str) -> list[Any]:
        from clearml import Task

        task = Task.get_task(task_id=task_id)
        return self.task_output_models(task)

    def list_tasks(self, project: str, tags: list[str] | None = None) -> list[Any]:
        from clearml import Task

        kwargs: dict[str, Any] = {"project_name": project, "allow_archived": False}
        if tags:
            kwargs["tags"] = tags
        return list(Task.get_tasks(**kwargs) or [])

    def task_metrics(self, task: Any) -> dict[str, float]:
        getter = getattr(task, "get_last_scalar_metrics", None)
        if callable(getter):
            return flatten_scalars(getter())
        return {}

    def task_output_models(self, task: Any) -> list[Any]:
        models = getattr(task, "models", None)
        if models is None:
            return []
        if isinstance(models, dict):
            outputs = models.get("output") or models.get("Output") or []
            return list(outputs)
        getter = getattr(models, "get", None)
        if callable(getter):
            return list(getter("output") or [])
        return []

    def apply_tags(self, model: Any, tags: list[str]) -> list[str]:
        current = [str(tag) for tag in (getattr(model, "tags", None) or [])]
        merged = list(dict.fromkeys([*current, *tags]))
        # ClearML Model exposes a writable ``tags`` list (not add_tags).
        try:
            model.tags = merged
            return merged
        except Exception:
            pass
        if hasattr(model, "add_tags") and callable(model.add_tags):
            model.add_tags(tags)
        elif hasattr(model, "set_tags") and callable(model.set_tags):
            model.set_tags(merged)
        return merged

    def publish(self, model: Any) -> None:
        publisher = getattr(model, "publish", None)
        if callable(publisher):
            publisher()


def promote_model(
    *,
    model_id: str | None = None,
    task_id: str | None = None,
    project: str | None = None,
    metric: str = "test_accuracy",
    higher_is_better: bool | None = None,
    tags: list[str] | None = None,
    publish: bool = True,
    compare_tag: str | None = None,
    registry: ModelRegistry | None = None,
) -> PromoteResult:
    """Tag and optionally publish a ClearML model.

    Resolution order: ``model_id`` → first output model on ``task_id`` →
    best task in ``project`` by ``metric``.
    """
    store = registry or ClearMLRegistry()
    label_tags = list(tags or ["production"])
    selected_metric = None
    selected_value = None
    selected_task_id = task_id
    model: Any

    if model_id:
        model = store.get_model(model_id)
    elif task_id:
        outputs = store.models_for_task(task_id)
        if not outputs:
            raise ValueError(f"Task {task_id!r} has no output models to promote.")
        model = outputs[0]
    elif project:
        query_tags = [compare_tag] if compare_tag else None
        tasks = store.list_tasks(project, tags=query_tags)
        if not tasks:
            raise ValueError(f"No tasks found in project {project!r}.")
        model, selected_task_id, selected_value = _best_model_for_metric(
            tasks, store, metric, higher_is_better
        )
        selected_metric = metric
    else:
        raise ValueError("Provide --model-id, --task-id, or --experiment/--project.")

    resolved_id = str(getattr(model, "id", None) or model_id or "")
    resolved_name = getattr(model, "name", None)
    applied = store.apply_tags(model, label_tags)
    if publish:
        store.publish(model)

    result = PromoteResult(
        model_id=resolved_id,
        model_name=str(resolved_name) if resolved_name else None,
        task_id=selected_task_id,
        tags=applied,
        published=publish,
        metric=selected_metric,
        metric_value=selected_value,
    )
    print(
        f"Promoted model {result.model_id}"
        + (f" ({result.model_name})" if result.model_name else "")
        + f"  published={result.published}  tags={','.join(result.tags)}"
    )
    if result.metric and result.metric_value is not None:
        print(f"  selected_by {result.metric}={result.metric_value:.4f}")
    return result


def metric_higher_is_better(metric: str, explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    key = metric.rsplit("/", 1)[-1]
    return key not in LOWER_IS_BETTER


def _best_model_for_metric(
    tasks: list[Any],
    store: ModelRegistry,
    metric: str,
    higher_is_better: bool | None,
) -> tuple[Any, str | None, float]:
    maximize = metric_higher_is_better(metric, higher_is_better)
    ranked: list[tuple[float, Any, str | None]] = []
    for task in tasks:
        metrics = store.task_metrics(task)
        value = _lookup_metric(metrics, metric)
        if value is None:
            continue
        outputs = store.task_output_models(task)
        if not outputs:
            continue
        task_id = str(getattr(task, "id", None) or getattr(task, "task_id", "") or "")
        ranked.append((value, outputs[0], task_id or None))
    if not ranked:
        raise ValueError(f"No tasks in the project exposed metric {metric!r} with an output model.")
    ranked.sort(key=lambda item: item[0], reverse=maximize)
    value, model, task_id = ranked[0]
    return model, task_id, value


def _lookup_metric(metrics: dict[str, float], name: str) -> float | None:
    if name in metrics:
        return float(metrics[name])
    suffix = name.rsplit("/", 1)[-1]
    if suffix in metrics:
        return float(metrics[suffix])
    for key, value in metrics.items():
        if key.endswith(name) or key.endswith(suffix):
            return float(value)
    return None
