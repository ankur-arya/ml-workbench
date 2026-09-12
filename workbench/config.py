"""YAML + CLI experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EXPERIMENT = "sklearn-workbench"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


@dataclass
class TrainConfig:
    experiment: str = DEFAULT_EXPERIMENT
    dataset: str = "iris"
    model: str = "random_forest"
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    tracking_uri: str = DEFAULT_TRACKING_URI
    register_model: bool = True
    run_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def registered_model_name(self) -> str | None:
        if not self.register_model:
            return None
        return f"{self.dataset}-{self.model}"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the root")
    return data


def config_from_mapping(data: dict[str, Any]) -> TrainConfig:
    known = {key: value for key, value in data.items() if key in TrainConfig.__dataclass_fields__}
    params = dict(known.pop("params", {}) or {})
    cfg = TrainConfig(**known)
    cfg.params = params
    return cfg


def merge_config(base: TrainConfig, **overrides: Any) -> TrainConfig:
    cleaned = {key: value for key, value in overrides.items() if value is not None}
    params = dict(base.params)
    extra_params = cleaned.pop("params", None)
    if extra_params:
        params.update(extra_params)
    updated = replace(base, **cleaned, params=params)
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
