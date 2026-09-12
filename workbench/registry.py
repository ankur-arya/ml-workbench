"""Model registry: candidate → staging → production, with an audit trail."""

from __future__ import annotations

from typing import Any

from workbench.ranking import compare_experiment
from workbench.store import Store

STAGES = ("candidate", "staging", "production", "archived")
ALIASES = {"champion": "production", "challenger": "staging"}


def normalize_stage(stage: str) -> str:
    key = stage.strip().lower()
    key = ALIASES.get(key, key)
    if key not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Choose one of: {', '.join(STAGES)}")
    return key


def register_run(store: Store, run: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    dataset = (run.get("dataset") or {}).get("slug") or "model"
    name = model_name or f"{dataset}-{run['model_name']}"
    return store.add_model_version(name, run["id"], stage="candidate")


def promote(
    store: Store,
    *,
    run_id: str | None = None,
    version_id: str | None = None,
    experiment_id: str | None = None,
    model_name: str | None = None,
    stage: str = "production",
    note: str,
    actor: str = "local",
) -> dict[str, Any]:
    """Move a version to ``stage``. Production is exclusive per registered model."""
    cleaned_note = (note or "").strip()
    if len(cleaned_note) < 4:
        raise ValueError("A promotion note is required (at least 4 characters).")
    target_stage = normalize_stage(stage)
    actor_name = (actor or "local").strip() or "local"

    version: dict[str, Any]
    if version_id:
        version = store.get_version(version_id)
    elif run_id:
        run = store.get_run(run_id)
        existing = store.versions_for_run(run_id)
        if existing:
            version = existing[0]
        else:
            version = register_run(store, run, model_name)
    elif experiment_id:
        experiment = store.get_experiment(experiment_id)
        runs = store.list_runs(experiment_id)
        comparison = compare_experiment(experiment, runs)
        winner = comparison["recommended"]
        if winner is None:
            raise ValueError("No recommended winner to promote. Check the leaderboard and constraints.")
        return promote(
            store,
            run_id=winner["id"],
            model_name=model_name,
            stage=target_stage,
            note=cleaned_note,
            actor=actor_name,
        )
    else:
        raise ValueError("Provide a run_id, version_id, or experiment_id.")

    from_stage = version["stage"]
    model_id = version["model_id"]
    if target_stage in {"production", "staging"}:
        store.archive_stage(model_id, target_stage, except_version=version["id"])
    store.set_version_stage(version["id"], target_stage)
    promotion = store.add_promotion(
        version_id=version["id"],
        from_stage=from_stage,
        to_stage=target_stage,
        note=cleaned_note,
        actor=actor_name,
    )
    updated = store.get_version(version["id"])
    model = store.get_registered_model(model_id)
    run = store.get_run(updated["run_id"])
    return {
        "model": model,
        "version": updated,
        "promotion": promotion,
        "run": {"id": run["id"], "name": run["name"], "experiment_id": run["experiment_id"]},
        "alias": "champion" if target_stage == "production" else ("challenger" if target_stage == "staging" else None),
    }
