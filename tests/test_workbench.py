"""Catalog and config smoke tests."""

from __future__ import annotations

import pytest

from workbench.config import parse_param_overrides
from workbench.datasets import list_datasets, load_dataset
from workbench.models import build_model, list_models


def test_datasets_and_models_are_consistent() -> None:
    assert set(list_datasets()) == {"iris", "wine", "breast_cancer", "diabetes"}
    iris = load_dataset("iris")
    assert iris.task == "classification"
    assert iris.X.shape[0] == 150
    diabetes = load_dataset("diabetes")
    assert diabetes.task == "regression"
    assert "random_forest" in list_models("classification")
    assert "ridge" in list_models("regression")
    with pytest.raises(ValueError):
        load_dataset("not-a-dataset")


def test_param_overrides() -> None:
    parsed = parse_param_overrides(["n_estimators=50", "max_depth=3", "bootstrap=true"])
    assert parsed == {"n_estimators": 50, "max_depth": 3, "bootstrap": True}


def test_build_model_rejects_wrong_task() -> None:
    with pytest.raises(ValueError, match="not valid"):
        build_model("ridge", "classification")
