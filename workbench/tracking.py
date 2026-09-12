"""ClearML tracking helpers (Task, Logger, OutputModel, Dataset).

The default backend talks to a ClearML Server (local docker-compose or the
hosted app.clear.ml workspace). Tests and ``WORKBENCH_TRACKER=memory`` use an
in-process recorder so CI does not need a server.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from workbench.config import DEFAULT_WEB_HOST


MEMORY_TRACKER = "memory"
CLEARML_TRACKER = "clearml"


def tracker_backend(explicit: str | None = None) -> str:
    raw = (explicit or os.environ.get("WORKBENCH_TRACKER") or CLEARML_TRACKER).strip().lower()
    if raw in {MEMORY_TRACKER, "none", "null", "off"}:
        return MEMORY_TRACKER
    return CLEARML_TRACKER


def is_offline(explicit: bool | None = None) -> bool:
    if explicit is True:
        return True
    env = os.environ.get("CLEARML_OFFLINE_MODE", "").strip().lower()
    return env in {"1", "true", "yes"}


def webapp_url() -> str:
    return (
        os.environ.get("CLEARML_WEB_HOST")
        or os.environ.get("CLEARML_WEB_SERVER")
        or DEFAULT_WEB_HOST
    ).rstrip("/")


def stringify(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


@dataclass
class TrainResult:
    task_id: str
    task_name: str
    project: str
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    model_id: str | None = None
    model_name: str | None = None
    dataset_id: str | None = None
    offline: bool = False
    web_url: str | None = None


class TrackingSession(Protocol):
    project: str
    task_name: str
    task_id: str

    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None: ...
    def set_tags(self, tags: dict[str, str] | list[str]) -> None: ...
    def log_artifact(self, path: str | Path, artifact_path: str = "") -> None: ...
    def log_text(self, name: str, text: str, artifact_path: str = "") -> None: ...
    def log_table(self, title: str, rows: list[dict[str, Any]]) -> None: ...
    def register_model(
        self,
        path: str | Path,
        name: str,
        tags: list[str] | None = None,
        comment: str = "",
    ) -> str | None: ...
    def register_dataset(
        self,
        name: str,
        csv_path: str | Path,
        preview_path: str | Path | None = None,
    ) -> str | None: ...
    def close(self) -> None: ...

    @property
    def handle(self) -> TrainResult: ...


class MemorySession:
    """In-process tracker used by tests and ``WORKBENCH_TRACKER=memory``."""

    def __init__(
        self,
        project: str,
        task_name: str,
        tags: list[str] | dict[str, str] | None = None,
        offline: bool = True,
    ) -> None:
        self.project = project
        self.task_name = task_name
        self.task_id = f"mem-{uuid.uuid4().hex[:12]}"
        self.params: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.tags: dict[str, str] = _normalize_tags(tags)
        self.artifacts: list[str] = []
        self.tables: list[tuple[str, list[dict[str, Any]]]] = []
        self.model_id: str | None = None
        self.model_name: str | None = None
        self.dataset_id: str | None = None
        self.offline = offline
        self.closed = False

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update({key: stringify(value) for key, value in params.items()})

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        self.metrics.update({f"{prefix}{key}": float(value) for key, value in metrics.items()})

    def set_tags(self, tags: dict[str, str] | list[str]) -> None:
        self.tags.update(_normalize_tags(tags))

    def log_artifact(self, path: str | Path, artifact_path: str = "") -> None:
        self.artifacts.append(_artifact_key(path, artifact_path))

    def log_text(self, name: str, text: str, artifact_path: str = "") -> None:
        del text
        rel = f"{artifact_path}/{name}" if artifact_path else name
        self.artifacts.append(rel)

    def log_table(self, title: str, rows: list[dict[str, Any]]) -> None:
        self.tables.append((title, list(rows)))

    def register_model(
        self,
        path: str | Path,
        name: str,
        tags: list[str] | None = None,
        comment: str = "",
    ) -> str | None:
        del comment
        self.model_name = name
        self.model_id = f"model-{uuid.uuid4().hex[:8]}"
        self.artifacts.append(_artifact_key(path, "model"))
        if tags:
            self.set_tags(tags)
        return self.model_id

    def register_dataset(
        self,
        name: str,
        csv_path: str | Path,
        preview_path: str | Path | None = None,
    ) -> str | None:
        del name, csv_path, preview_path
        self.dataset_id = f"ds-{uuid.uuid4().hex[:8]}"
        return self.dataset_id

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> MemorySession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def handle(self) -> TrainResult:
        return TrainResult(
            task_id=self.task_id,
            task_name=self.task_name,
            project=self.project,
            metrics=dict(self.metrics),
            params=dict(self.params),
            tags=dict(self.tags),
            artifacts=list(self.artifacts),
            model_id=self.model_id,
            model_name=self.model_name,
            dataset_id=self.dataset_id,
            offline=self.offline,
            web_url=None,
        )


class ClearMLSession:
    """Live ClearML Task: params, scalars, artifacts, OutputModel, Dataset."""

    def __init__(
        self,
        project: str,
        task_name: str,
        tags: list[str] | dict[str, str] | None = None,
        offline: bool = False,
        task_type: str = "training",
    ) -> None:
        from clearml import Task

        self.offline = offline or is_offline()
        if self.offline:
            Task.set_offline(offline_mode=True)

        task_types = getattr(Task, "TaskTypes", None)
        resolved_type = task_type
        if task_types is not None:
            resolved_type = getattr(task_types, task_type, task_type)

        self.task = Task.init(
            project_name=project,
            task_name=task_name,
            task_type=resolved_type,
            reuse_last_task_id=False,
            auto_connect_frameworks={"matplotlib": True, "sklearn": False},
        )
        self.logger = self.task.get_logger()
        self.project = project
        self.task_name = task_name
        self.task_id = str(getattr(self.task, "id", None) or task_name)
        self.params: dict[str, str] = {}
        self.metrics: dict[str, float] = {}
        self.tags: dict[str, str] = _normalize_tags(tags)
        self.artifacts: list[str] = []
        self.tables: list[tuple[str, list[dict[str, Any]]]] = []
        self.model_id: str | None = None
        self.model_name: str | None = None
        self.dataset_id: str | None = None
        self.closed = False
        if self.tags:
            self._apply_task_tags(list(self.tags))

    def log_params(self, params: dict[str, Any]) -> None:
        payload = {key: stringify(value) for key, value in params.items()}
        self.params.update(payload)
        connect = getattr(self.task, "connect", None)
        if callable(connect):
            connect(payload, name="General")

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        for key, value in metrics.items():
            name = f"{prefix}{key}"
            number = float(value)
            self.metrics[name] = number
            self.logger.report_single_value(name=name, value=number)
            self.logger.report_scalar(title="metrics", series=name, iteration=0, value=number)

    def set_tags(self, tags: dict[str, str] | list[str]) -> None:
        normalized = _normalize_tags(tags)
        self.tags.update(normalized)
        self._apply_task_tags(list(normalized))

    def log_artifact(self, path: str | Path, artifact_path: str = "") -> None:
        path = Path(path)
        key = _artifact_key(path, artifact_path)
        self.artifacts.append(key)
        artifact_name = key.replace("/", "__")
        self.task.upload_artifact(artifact_name, artifact_object=str(path))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
            self.logger.report_image(
                title=artifact_path or "plots",
                series=path.stem,
                iteration=0,
                local_path=str(path),
            )

    def log_text(self, name: str, text: str, artifact_path: str = "") -> None:
        rel = f"{artifact_path}/{name}" if artifact_path else name
        self.artifacts.append(rel)
        self.task.upload_artifact(rel.replace("/", "__"), artifact_object=text)

    def log_table(self, title: str, rows: list[dict[str, Any]]) -> None:
        self.tables.append((title, list(rows)))
        if not rows:
            return
        try:
            import pandas as pd

            frame = pd.DataFrame(rows)
            self.logger.report_table(title=title, series="results", iteration=0, table_plot=frame)
        except Exception:
            self.task.upload_artifact(title, artifact_object=rows)

    def register_model(
        self,
        path: str | Path,
        name: str,
        tags: list[str] | None = None,
        comment: str = "",
    ) -> str | None:
        from clearml import OutputModel

        path = Path(path)
        output = OutputModel(
            task=self.task,
            name=name,
            framework="scikit-learn",
            tags=list(tags or []),
            comment=comment or None,
        )
        output.update_weights(weights_filename=str(path))
        self.model_name = name
        self.model_id = str(getattr(output, "id", None) or name)
        self.artifacts.append(_artifact_key(path, "model"))
        if tags:
            self.set_tags(tags)
        return self.model_id

    def register_dataset(
        self,
        name: str,
        csv_path: str | Path,
        preview_path: str | Path | None = None,
    ) -> str | None:
        if self.offline:
            self.dataset_id = None
            return None
        try:
            from clearml import Dataset

            dataset = Dataset.create(
                dataset_name=name,
                dataset_project=f"{self.project}/datasets",
            )
            dataset.add_files(str(csv_path))
            if preview_path:
                dataset.add_files(str(preview_path))
            dataset.upload()
            dataset.finalize()
            self.dataset_id = str(getattr(dataset, "id", None) or name)
            connect = getattr(self.task, "connect", None)
            if callable(connect):
                try:
                    connect(dataset)
                except Exception:
                    pass
            return self.dataset_id
        except Exception as exc:
            self.log_text("dataset_register_error.txt", str(exc), artifact_path="reports")
            return None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        flush = getattr(self.logger, "flush", None)
        if callable(flush):
            flush()
        closer = getattr(self.task, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> ClearMLSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _apply_task_tags(self, tags: list[str]) -> None:
        add = getattr(self.task, "add_tags", None)
        if callable(add):
            add(tags)
            return
        setter = getattr(self.task, "set_tags", None)
        if callable(setter):
            setter(tags)

    @property
    def handle(self) -> TrainResult:
        url = None if self.offline else f"{webapp_url()}/projects"
        return TrainResult(
            task_id=self.task_id,
            task_name=self.task_name,
            project=self.project,
            metrics=dict(self.metrics),
            params=dict(self.params),
            tags=dict(self.tags),
            artifacts=list(self.artifacts),
            model_id=self.model_id,
            model_name=self.model_name,
            dataset_id=self.dataset_id,
            offline=self.offline,
            web_url=url,
        )


def open_session(
    project: str,
    task_name: str,
    *,
    offline: bool = False,
    tags: list[str] | dict[str, str] | None = None,
    task_type: str = "training",
    backend: str | None = None,
) -> TrackingSession:
    if tracker_backend(backend) == MEMORY_TRACKER:
        return MemorySession(project, task_name, tags=tags, offline=True)
    return ClearMLSession(
        project,
        task_name,
        tags=tags,
        offline=offline,
        task_type=task_type,
    )


def flatten_scalars(nested: dict[str, Any] | None) -> dict[str, float]:
    """Flatten ``Task.get_last_scalar_metrics()`` into ``name -> last value``."""
    flat: dict[str, float] = {}
    for title, series_map in (nested or {}).items():
        if isinstance(series_map, (int, float)):
            flat[str(title)] = float(series_map)
            continue
        if not isinstance(series_map, dict):
            continue
        if "last" in series_map and isinstance(series_map["last"], (int, float)):
            flat[str(title)] = float(series_map["last"])
            continue
        for series, stats in series_map.items():
            if isinstance(stats, dict) and "last" in stats:
                value = float(stats["last"])
            elif isinstance(stats, (int, float)):
                value = float(stats)
            else:
                continue
            flat[str(series)] = value
            flat[f"{title}/{series}"] = value
    return flat


def _normalize_tags(tags: list[str] | dict[str, str] | None) -> dict[str, str]:
    if not tags:
        return {}
    if isinstance(tags, dict):
        return {str(key): stringify(value) for key, value in tags.items()}
    return {str(tag): "true" for tag in tags}


def _artifact_key(path: str | Path, artifact_path: str) -> str:
    name = Path(path).name
    return f"{artifact_path}/{name}" if artifact_path else name
