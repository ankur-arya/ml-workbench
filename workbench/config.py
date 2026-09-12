"""YAML + CLI experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EXPERIMENT = "sklearn-workbench"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_WEB_HOST = "http://localhost:8080"
DEFAULT_API_HOST = "http://localhost:8008"
DEFAULT_FILES_HOST = "http://localhost:8081"


@dataclass
class TrainConfig:
    """One training run: dataset + model + ClearML project (experiment).

    ``experiment`` is the ClearML *project* name so related tasks show up
    together in the WebApp.
    """

    experiment: str = DEFAULT_EXPERIMENT
    dataset: str = "iris"
    model: str = "random_forest"
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    register_model: bool = True
    register_dataset: bool = False
    offline: bool = False
    run_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @property
    def project(self) -> str:
        return self.experiment

    @property
    def registered_model_name(self) -> str | None:
        if not self.register_model:
            return None
        return f"{self.dataset}-{self.model}"


@dataclass
class CompareConfig:
    experiment: str = DEFAULT_EXPERIMENT
    datasets: list[str] = field(default_factory=lambda: ["iris", "wine", "diabetes"])
    models: list[str] | None = None
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    register_model: bool = True
    register_dataset: bool = False
    offline: bool = False
    tags: list[str] = field(default_factory=list)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the root")
    return data


def config_from_mapping(data: dict[str, Any]) -> TrainConfig:
    known = {key: value for key, value in data.items() if key in TrainConfig.__dataclass_fields__}
    params = dict(known.pop("params", {}) or {})
    tags = list(known.pop("tags", []) or [])
    cfg = TrainConfig(**known)
    cfg.params = params
    cfg.tags = tags
    return cfg


def compare_config_from_mapping(data: dict[str, Any]) -> CompareConfig:
    known = {key: value for key, value in data.items() if key in CompareConfig.__dataclass_fields__}
    if "dataset" in data and "datasets" not in known:
        value = data["dataset"]
        known["datasets"] = [value] if isinstance(value, str) else list(value)
    if "model" in data and "models" not in known:
        value = data["model"]
        known["models"] = [value] if isinstance(value, str) else list(value)
    tags = list(known.pop("tags", []) or [])
    cfg = CompareConfig(**known)
    cfg.tags = tags
    return cfg


def merge_config(base: TrainConfig, **overrides: Any) -> TrainConfig:
    cleaned = {key: value for key, value in overrides.items() if value is not None}
    params = dict(base.params)
    extra_params = cleaned.pop("params", None)
    if extra_params:
        params.update(extra_params)
    tags = list(base.tags)
    extra_tags = cleaned.pop("tags", None)
    if extra_tags:
        tags.extend(tag for tag in extra_tags if tag not in tags)
    updated = replace(base, **cleaned, params=params, tags=tags)
    return updated


def parse_param_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` CLI pairs with light type coercion."""
    parsed: dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Expected KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        parsed[key.strip()] = _coerce(value.strip())
    return parsed


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        return float(value)
    except ValueError:
        return value
