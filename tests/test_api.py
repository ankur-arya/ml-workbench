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
