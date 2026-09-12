"""Promote records who / when / why and exclusive production."""

from __future__ import annotations

import pytest

from workbench.registry import promote, register_run
from workbench.store import Store


def _ready_run(store: Store) -> dict:
    experiment = store.create_experiment(name="promo", primary_metric="test_accuracy")
    run = store.create_run(
        experiment_id=experiment["id"],
        name="iris-random_forest",
        model_name="random_forest",
        dataset_id="iris",
        task="classification",
        status="succeeded",
    )
    store.set_metrics(run["id"], {"test_accuracy": 0.96})
    return store.get_run(run["id"])


def test_promote_requires_a_note(store: Store) -> None:
    run = _ready_run(store)
    with pytest.raises(ValueError, match="note"):
        promote(store, run_id=run["id"], note="no")


def test_promote_to_production_and_audit(store: Store) -> None:
    run = _ready_run(store)
    result = promote(store, run_id=run["id"], stage="production", note="Best holdout accuracy", actor="ankur")
    assert result["version"]["stage"] == "production"
    assert result["alias"] == "champion"
    assert result["promotion"]["note"] == "Best holdout accuracy"
    assert result["promotion"]["actor"] == "ankur"
    assert result["promotion"]["from_stage"] == "candidate"
    assert result["run"]["id"] == run["id"]
    registry = store.list_registry()
    assert registry[0]["production"]["run_id"] == run["id"]


def test_promote_archives_previous_production(store: Store) -> None:
    first = _ready_run(store)
    second_exp = store.create_experiment(name="promo-2", primary_metric="test_accuracy")
    second = store.create_run(
        experiment_id=second_exp["id"],
        name="iris-random_forest-v2",
        model_name="random_forest",
        dataset_id="iris",
        task="classification",
        status="succeeded",
    )
    store.set_metrics(second["id"], {"test_accuracy": 0.99})
    promote(store, run_id=first["id"], stage="production", note="First champion on iris")
    promote(
        store,
        run_id=second["id"],
        stage="production",
        note="Beats the previous champion",
        model_name="iris-random_forest",
    )
    versions = store.list_registry()[0]["versions"]
    stages = {item["run_id"]: item["stage"] for item in versions}
    assert stages[second["id"]] == "production"
    assert stages[first["id"]] == "archived"


def test_promote_experiment_winner(store: Store) -> None:
    experiment = store.create_experiment(name="bakeoff", primary_metric="test_accuracy", maximize=True)
    weak = store.create_run(
        experiment_id=experiment["id"],
        name="iris-lr",
        model_name="logistic_regression",
        dataset_id="iris",
        task="classification",
        status="succeeded",
    )
    strong = store.create_run(
        experiment_id=experiment["id"],
        name="iris-rf",
        model_name="random_forest",
        dataset_id="iris",
        task="classification",
        status="succeeded",
    )
    store.set_metrics(weak["id"], {"test_accuracy": 0.8})
    store.set_metrics(strong["id"], {"test_accuracy": 0.95})
    result = promote(store, experiment_id=experiment["id"], note="Leaderboard winner")
    assert result["run"]["id"] == strong["id"]
    assert result["version"]["stage"] == "production"


def test_register_run_creates_candidate(store: Store) -> None:
    run = _ready_run(store)
    version = register_run(store, run)
    assert version["stage"] == "candidate"
    assert version["run_id"] == run["id"]
