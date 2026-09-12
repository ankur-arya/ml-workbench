"""YAML + CLI experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EXPERIMENT = "sklearn-workbench"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_PRIMARY_CLASSIFICATION = "test_accuracy"
DEFAULT_PRIMARY_REGRESSION = "test_r2"


@dataclass
class Constraint:
    """Keep a run only if ``metrics[metric] >= min`` (or ``<= max``)."""

    metric: str
    min: float | None = None
    max: float | None = None

    def allows(self, metrics: dict[str, float]) -> bool:
        value = metrics.get(self.metric)
        if value is None:
            return False
        if self.min is not None and value < self.min:
            return False
        if self.max is not None and value > self.max:
            return False
        return True


@dataclass
class ModelSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """One training run: dataset + model + experiment name."""

    experiment: str = DEFAULT_EXPERIMENT
    description: str = ""
    dataset: str = "iris"
    model: str = "random_forest"
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    register_model: bool = True
    run_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    primary_metric: str | None = None
    maximize: bool | None = None

    @property
    def registered_model_name(self) -> str | None:
        if not self.register_model:
            return None
        return f"{self.dataset}-{self.model}"


@dataclass
class CompareConfig:
    experiment: str = DEFAULT_EXPERIMENT
    description: str = ""
    datasets: list[str] = field(default_factory=lambda: ["iris"])
    models: list[ModelSpec] | None = None
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    register_model: bool = True
    tags: list[str] = field(default_factory=list)
    primary_metric: str | None = None
    maximize: bool | None = None
    constraints: list[Constraint] = field(default_factory=list)


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


def _parse_model_specs(raw: Any) -> list[ModelSpec] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return [ModelSpec(name=raw)]
    if not isinstance(raw, list):
        raise ValueError("models must be a list of names or {name, params} objects")
    specs: list[ModelSpec] = []
    for item in raw:
        if isinstance(item, str):
            specs.append(ModelSpec(name=item))
        elif isinstance(item, dict) and item.get("name"):
            specs.append(ModelSpec(name=str(item["name"]), params=dict(item.get("params") or {})))
        else:
            raise ValueError(f"Invalid model spec: {item!r}")
    return specs


def _parse_constraints(raw: Any) -> list[Constraint]:
    if not raw:
        return []
    out: list[Constraint] = []
    for item in raw:
        if not isinstance(item, dict) or "metric" not in item:
            raise ValueError(f"Invalid constraint: {item!r}")
        out.append(
            Constraint(
                metric=str(item["metric"]),
                min=item.get("min"),
                max=item.get("max"),
            )
        )
    return out


def compare_config_from_mapping(data: dict[str, Any]) -> CompareConfig:
    payload = dict(data)
    if "dataset" in payload and "datasets" not in payload:
        value = payload.pop("dataset")
        payload["datasets"] = [value] if isinstance(value, str) else list(value)
    if "model" in payload and "models" not in payload:
        payload["models"] = payload.pop("model")
    models = _parse_model_specs(payload.pop("models", None))
    constraints = _parse_constraints(payload.pop("constraints", None))
    known = {key: value for key, value in payload.items() if key in CompareConfig.__dataclass_fields__}
    tags = list(known.pop("tags", []) or [])
    cfg = CompareConfig(**known)
    cfg.models = models
    cfg.tags = tags
    cfg.constraints = constraints
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
    return replace(base, **cleaned, params=params, tags=tags)


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


def default_primary_for_task(task: str) -> tuple[str, bool]:
    if task == "regression":
        return DEFAULT_PRIMARY_REGRESSION, True
    return DEFAULT_PRIMARY_CLASSIFICATION, True
