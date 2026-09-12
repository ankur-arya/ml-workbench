from __future__ import annotations

from fastapi.testclient import TestClient

from workbench.api import create_app
from workbench.store import Store


def test_health_and_demo_promote(store: Store) -> None:
    client = TestClient(create_app(store))
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    created = client.post("/api/demo?background=false")
    assert created.status_code == 200
    body = created.json()
    assert body["status"] == "completed"
    assert body["compare"]["recommended"] is not None
    winner_id = body["compare"]["recommended"]["id"]

    promoted = client.post(
        "/api/promote",
        json={"run_id": winner_id, "stage": "production", "note": "Demo winner to production", "actor": "pytest"},
    )
    assert promoted.status_code == 200
    assert promoted.json()["version"]["stage"] == "production"

    registry = client.get("/api/registry").json()
    assert registry
    assert registry[0]["production"]["run_id"] == winner_id

    experiments = client.get("/api/experiments").json()
    assert experiments[0]["winner_id"] == winner_id


def test_promote_api_returns_note_validation_detail(store: Store) -> None:
    client = TestClient(create_app(store))
    experiment = store.create_experiment(name="short-note", primary_metric="test_accuracy")
    run = store.create_run(
        experiment_id=experiment["id"],
        name="iris-rf",
        model_name="random_forest",
        dataset_id="iris",
        task="classification",
        status="succeeded",
    )
    store.set_metrics(run["id"], {"test_accuracy": 0.96})
    rejected = client.post(
        "/api/promote",
        json={"run_id": run["id"], "stage": "production", "note": "no", "actor": "pytest"},
    )
    assert rejected.status_code == 400
    assert rejected.json()["detail"] == "A promotion note is required (at least 4 characters)."
    assert "body stream already read" not in rejected.json()["detail"]
