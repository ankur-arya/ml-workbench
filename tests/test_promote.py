from __future__ import annotations

import pytest

from workbench.promote import PromoteResult, metric_higher_is_better, promote_model


class FakeModel:
    def __init__(self, model_id: str, name: str, tags: list[str] | None = None) -> None:
        self.id = model_id
        self.name = name
        self.tags = list(tags or [])
        self.published = False

    def add_tags(self, tags: list[str]) -> None:
        self.tags.extend(tags)

    def publish(self) -> None:
        self.published = True


class FakeTask:
    def __init__(self, task_id: str, metrics: dict, model: FakeModel | None) -> None:
        self.id = task_id
        self._metrics = metrics
        self.models = {"output": [model] if model else []}

    def get_last_scalar_metrics(self) -> dict:
        return {"metrics": {key: {"last": value} for key, value in self._metrics.items()}}


class FakeRegistry:
    def __init__(self, models: dict[str, FakeModel], tasks: list[FakeTask]) -> None:
        self.models = models
        self.tasks = tasks
        self.published: list[str] = []

    def get_model(self, model_id: str) -> FakeModel:
        return self.models[model_id]

    def models_for_task(self, task_id: str) -> list[FakeModel]:
        for task in self.tasks:
            if task.id == task_id:
                return list(task.models["output"])
        return []

    def list_tasks(self, project: str, tags: list[str] | None = None) -> list[FakeTask]:
        del project, tags
        return list(self.tasks)

    def task_metrics(self, task: FakeTask) -> dict[str, float]:
        from workbench.tracking import flatten_scalars

        return flatten_scalars(task.get_last_scalar_metrics())

    def task_output_models(self, task: FakeTask) -> list[FakeModel]:
        return list(task.models["output"])

    def apply_tags(self, model: FakeModel, tags: list[str]) -> list[str]:
        model.add_tags(tags)
        return list(model.tags)

    def publish(self, model: FakeModel) -> None:
        model.publish()
        self.published.append(model.id)


def test_promote_by_model_id() -> None:
    model = FakeModel("m1", "iris-rf")
    registry = FakeRegistry({"m1": model}, [])
    result = promote_model(
        model_id="m1",
        tags=["production", "champion"],
        publish=True,
        registry=registry,
    )
    assert isinstance(result, PromoteResult)
    assert result.published is True
    assert model.published is True
    assert "production" in result.tags
    assert "champion" in result.tags


def test_promote_best_in_project() -> None:
    weak = FakeModel("weak", "iris-logreg")
    strong = FakeModel("strong", "iris-rf")
    tasks = [
        FakeTask("t1", {"test_accuracy": 0.80}, weak),
        FakeTask("t2", {"test_accuracy": 0.97}, strong),
    ]
    registry = FakeRegistry({"weak": weak, "strong": strong}, tasks)
    result = promote_model(
        project="sklearn-workbench",
        metric="test_accuracy",
        tags=["production"],
        registry=registry,
    )
    assert result.model_id == "strong"
    assert result.metric_value == pytest.approx(0.97)
    assert strong.published is True
    assert weak.published is False


def test_promote_minimizes_rmse() -> None:
    worse = FakeModel("worse", "diabetes-ridge")
    better = FakeModel("better", "diabetes-rf")
    tasks = [
        FakeTask("t1", {"test_rmse": 55.0}, worse),
        FakeTask("t2", {"test_rmse": 40.0}, better),
    ]
    registry = FakeRegistry({}, tasks)
    result = promote_model(
        project="sklearn-workbench",
        metric="test_rmse",
        higher_is_better=False,
        tags=["production"],
        registry=registry,
    )
    assert result.model_id == "better"


def test_metric_direction() -> None:
    assert metric_higher_is_better("test_accuracy") is True
    assert metric_higher_is_better("test_rmse") is False
    assert metric_higher_is_better("test_rmse", True) is True


def test_promote_requires_selector() -> None:
    with pytest.raises(ValueError, match="Provide"):
        promote_model(registry=FakeRegistry({}, []))


def test_promote_task_without_models() -> None:
    registry = FakeRegistry({}, [FakeTask("empty", {"test_accuracy": 1.0}, None)])
    with pytest.raises(ValueError, match="no output models"):
        promote_model(task_id="empty", registry=registry)
