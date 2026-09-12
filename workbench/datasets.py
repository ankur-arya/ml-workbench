"""Load bundled scikit-learn datasets and user tabular files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
)

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    task: TaskType
    frame: pd.DataFrame
    features: list[str]
    target: str
    target_names: list[str] | None
    description: str
    source: str = "builtin"
    path: str | None = None

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.features]

    @property
    def y(self) -> pd.Series:
        return self.frame[self.target]


def _from_sklearn(
    name: str,
    task: TaskType,
    loader: Callable[..., object],
    description: str,
) -> DatasetBundle:
    bunch = loader(as_frame=True)
    frame = bunch.frame.copy()
    target = bunch.target.name or "target"
    features = [column for column in frame.columns if column != target]
    target_names = list(getattr(bunch, "target_names", [])) or None
    return DatasetBundle(
        name=name,
        task=task,
        frame=frame,
        features=features,
        target=target,
        target_names=target_names,
        description=description,
        source="builtin",
    )


DATASET_META: dict[str, dict[str, str]] = {
    "iris": {
        "task": "classification",
        "description": "3-class flower taxonomy (150 rows, 4 numeric features).",
    },
    "wine": {
        "task": "classification",
        "description": "3-class wine cultivar recognition (178 rows, 13 features).",
    },
    "breast_cancer": {
        "task": "classification",
        "description": "Binary tumor diagnosis (569 rows, 30 features).",
    },
    "diabetes": {
        "task": "regression",
        "description": "Disease progression regression (442 rows, 10 features).",
    },
}

DATASET_LOADERS: dict[str, Callable[[], DatasetBundle]] = {
    "iris": lambda: _from_sklearn("iris", "classification", load_iris, DATASET_META["iris"]["description"]),
    "wine": lambda: _from_sklearn("wine", "classification", load_wine, DATASET_META["wine"]["description"]),
    "breast_cancer": lambda: _from_sklearn(
        "breast_cancer",
        "classification",
        load_breast_cancer,
        DATASET_META["breast_cancer"]["description"],
    ),
    "diabetes": lambda: _from_sklearn(
        "diabetes",
        "regression",
        load_diabetes,
        DATASET_META["diabetes"]["description"],
    ),
}


def list_datasets() -> list[str]:
    return sorted(DATASET_LOADERS)


def load_dataset(name: str) -> DatasetBundle:
    key = name.strip().lower()
    if key not in DATASET_LOADERS:
        available = ", ".join(list_datasets())
        raise ValueError(f"Unknown dataset {name!r}. Choose one of: {available}")
    return DATASET_LOADERS[key]()


def load_tabular(
    path: str | Path,
    *,
    name: str,
    target: str,
    task: TaskType,
    description: str = "",
) -> DatasetBundle:
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} not in {path}")
    features = [column for column in frame.columns if column != target]
    target_names = None
    if task == "classification":
        target_names = [str(value) for value in sorted(frame[target].astype(str).unique())]
    return DatasetBundle(
        name=name,
        task=task,
        frame=frame,
        features=features,
        target=target,
        target_names=target_names,
        description=description or f"Tabular file ({frame.shape[0]} rows, {len(features)} features).",
        source="file",
        path=str(Path(path).resolve()),
    )


def snapshot_builtin(name: str) -> dict[str, object]:
    bundle = load_dataset(name)
    return {
        "slug": bundle.name,
        "name": bundle.name.replace("_", " ").title(),
        "source": "builtin",
        "task": bundle.task,
        "n_rows": int(bundle.frame.shape[0]),
        "n_features": len(bundle.features),
        "target": bundle.target,
        "feature_names": bundle.features,
        "target_names": bundle.target_names,
        "description": bundle.description,
        "path": None,
    }
