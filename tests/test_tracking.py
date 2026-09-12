from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from workbench.tracking import (
    ClearMLSession,
    MemorySession,
    flatten_scalars,
    open_session,
    tracker_backend,
)


def test_memory_session_records_artifacts(tmp_path: Path) -> None:
    session = MemorySession("proj", "run-1", tags=["candidate"])
    plot = tmp_path / "confusion_matrix.png"
    plot.write_bytes(b"png")
    session.log_params({"dataset": "iris"})
    session.log_metrics({"accuracy": 0.9}, prefix="test_")
    session.log_artifact(plot, artifact_path="plots")
    session.register_model(tmp_path / "model.joblib", "iris-rf", tags=["candidate"])
    session.close()
    handle = session.handle
    assert handle.params["dataset"] == "iris"
    assert handle.metrics["test_accuracy"] == 0.9
    assert "plots/confusion_matrix.png" in handle.artifacts
    assert handle.model_name == "iris-rf"
    assert session.closed is True


def test_open_session_honors_memory_backend(monkeypatch) -> None:
    monkeypatch.setenv("WORKBENCH_TRACKER", "memory")
    assert tracker_backend() == "memory"
    session = open_session("proj", "task")
    assert isinstance(session, MemorySession)
    session.close()


def test_flatten_scalars() -> None:
    nested = {
        "metrics": {"test_accuracy": {"last": 0.91, "min": 0.9, "max": 0.91}},
        "Summary": {"test_rmse": {"last": 12.5}},
    }
    flat = flatten_scalars(nested)
    assert flat["test_accuracy"] == 0.91
    assert flat["metrics/test_accuracy"] == 0.91
    assert flat["test_rmse"] == 12.5


def test_clearml_session_logs_and_registers(monkeypatch, tmp_path: Path) -> None:
    logger = MagicMock()
    task = MagicMock()
    task.id = "task-42"
    task.get_logger.return_value = logger

    task_cls = MagicMock()
    task_cls.init.return_value = task
    task_cls.TaskTypes.training = "training"
    task_cls.set_offline = MagicMock()

    output = MagicMock()
    output.id = "model-99"
    output_cls = MagicMock(return_value=output)

    fake = SimpleNamespace(Task=task_cls, OutputModel=output_cls, Dataset=MagicMock())
    monkeypatch.setitem(__import__("sys").modules, "clearml", fake)

    weights = tmp_path / "model.joblib"
    weights.write_bytes(b"joblib")
    image = tmp_path / "residuals.png"
    image.write_bytes(b"png")

    session = ClearMLSession("sklearn-workbench", "iris-ridge", tags=["iris"], offline=True)
    session.log_params({"dataset": "iris", "model": "ridge"})
    session.log_metrics({"r2": 0.4}, prefix="test_")
    session.log_artifact(image, artifact_path="plots")
    model_id = session.register_model(weights, "iris-ridge", tags=["candidate"])
    session.close()

    task_cls.set_offline.assert_called_once_with(offline_mode=True)
    task_cls.init.assert_called_once()
    kwargs = task_cls.init.call_args.kwargs
    assert kwargs["project_name"] == "sklearn-workbench"
    assert kwargs["task_name"] == "iris-ridge"
    logger.report_scalar.assert_called()
    logger.report_single_value.assert_called()
    task.upload_artifact.assert_called()
    output_cls.assert_called()
    output.update_weights.assert_called_once()
    assert model_id == "model-99"
    task.close.assert_called_once()
