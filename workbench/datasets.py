"""Load bundled scikit-learn datasets as named DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
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
    )


DATASET_LOADERS: dict[str, Callable[[], DatasetBundle]] = {
    "iris": lambda: _from_sklearn(
        "iris",
        "classification",
        load_iris,
        "3-class flower taxonomy (150 rows, 4 numeric features).",
    ),
    "wine": lambda: _from_sklearn(
        "wine",
        "classification",
        load_wine,
        "3-class wine cultivar recognition (178 rows, 13 features).",
    ),
    "breast_cancer": lambda: _from_sklearn(
        "breast_cancer",
        "classification",
        load_breast_cancer,
        "Binary tumor diagnosis (569 rows, 30 features).",
    ),
    "diabetes": lambda: _from_sklearn(
        "diabetes",
        "regression",
        load_diabetes,
        "Disease progression regression (442 rows, 10 features).",
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
