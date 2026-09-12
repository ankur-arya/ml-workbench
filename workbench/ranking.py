"""Compare-first ranking: leaderboard + recommended winner."""

from __future__ import annotations

from typing import Any

from workbench.config import Constraint, default_primary_for_task


def metric_value(run: dict[str, Any], key: str) -> float | None:
    metrics = run.get("metrics") or {}
    if key in metrics:
        return float(metrics[key])
    # Allow callers to pass "accuracy" and still match "test_accuracy".
    if not key.startswith("test_") and f"test_{key}" in metrics:
        return float(metrics[f"test_{key}"])
    return None


def constraints_from_raw(raw: list[dict[str, Any]] | None) -> list[Constraint]:
    out: list[Constraint] = []
    for item in raw or []:
        out.append(
            Constraint(
                metric=str(item["metric"]),
                min=item.get("min"),
                max=item.get("max"),
            )
        )
    return out


def passes_constraints(run: dict[str, Any], constraints: list[Constraint]) -> bool:
    metrics = {key: float(value) for key, value in (run.get("metrics") or {}).items()}
    return all(constraint.allows(metrics) for constraint in constraints)


def infer_primary(runs: list[dict[str, Any]], explicit: str | None, maximize: bool | None) -> tuple[str, bool]:
    if explicit:
        return explicit, True if maximize is None else bool(maximize)
    tasks = {run.get("task") for run in runs if run.get("task")}
    if tasks == {"regression"}:
        return default_primary_for_task("regression")
    return default_primary_for_task("classification")


def rank_runs(
    runs: list[dict[str, Any]],
    *,
    primary_metric: str,
    maximize: bool = True,
    constraints: list[Constraint] | None = None,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Return succeeded runs sorted by the primary metric, with rank metadata."""
    filtered: list[dict[str, Any]] = []
    for run in runs:
        if run.get("status") != "succeeded":
            continue
        if dataset:
            slug = (run.get("dataset") or {}).get("slug") or (run.get("dataset") or {}).get("id")
            if slug != dataset:
                continue
        filtered.append(run)

    eligible = [run for run in filtered if passes_constraints(run, constraints or [])]
    blocked = [run for run in filtered if run not in eligible]

    def sort_key(run: dict[str, Any]) -> tuple[int, float]:
        value = metric_value(run, primary_metric)
        if value is None:
            return (1, 0.0)
        return (0, -value if maximize else value)

    ordered = sorted(eligible, key=sort_key)
    best_value = metric_value(ordered[0], primary_metric) if ordered else None
    ranked: list[dict[str, Any]] = []
    for index, run in enumerate(ordered):
        value = metric_value(run, primary_metric)
        delta = None
        if value is not None and best_value is not None:
            delta = value - best_value
        ranked.append(
            {
                **run,
                "rank": index + 1,
                "primary_value": value,
                "delta_vs_best": delta,
                "is_recommended": index == 0 and value is not None,
                "constraint_failed": False,
                "missing_metric": value is None,
            }
        )

    for run in blocked:
        ranked.append(
            {
                **run,
                "rank": None,
                "primary_value": metric_value(run, primary_metric),
                "delta_vs_best": None,
                "is_recommended": False,
                "constraint_failed": True,
                "missing_metric": metric_value(run, primary_metric) is None,
            }
        )
    return ranked


def recommend(ranked: list[dict[str, Any]]) -> dict[str, Any] | None:
    for run in ranked:
        if run.get("is_recommended"):
            return run
    return None


def compare_experiment(
    experiment: dict[str, Any],
    runs: list[dict[str, Any]],
    *,
    dataset: str | None = None,
) -> dict[str, Any]:
    primary, maximize = infer_primary(runs, experiment.get("primary_metric"), experiment.get("maximize"))
    constraints = constraints_from_raw(experiment.get("constraints"))
    datasets = sorted(
        {
            (run.get("dataset") or {}).get("slug")
            for run in runs
            if (run.get("dataset") or {}).get("slug")
        }
    )
    tasks = sorted({run.get("task") for run in runs if run.get("task")})
    mixed_task = len(tasks) > 1

    leaderboard = rank_runs(
        runs,
        primary_metric=primary,
        maximize=maximize,
        constraints=constraints,
        dataset=dataset,
    )
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for slug in datasets:
        by_dataset[slug] = rank_runs(
            runs,
            primary_metric=primary,
            maximize=maximize,
            constraints=constraints,
            dataset=slug,
        )

    winner = None if (mixed_task and not dataset) else recommend(leaderboard)
    reason = None
    if winner:
        direction = "higher" if maximize else "lower"
        reason = (
            f"Best {primary.replace('_', ' ')} on "
            f"{dataset or 'this experiment'} ({direction} is better)."
        )
        if constraints:
            reason += " Passed configured constraints."
    elif mixed_task and not dataset:
        reason = "This experiment mixes classification and regression — pick a dataset facet."

    metric_keys = sorted({key for run in runs for key in (run.get("metrics") or {})})
    param_keys = sorted(
        {
            key
            for run in runs
            for key in (run.get("params") or {})
            if key.startswith("model__")
        }
    )

    return {
        "experiment_id": experiment["id"],
        "primary_metric": primary,
        "maximize": maximize,
        "datasets": datasets,
        "tasks": tasks,
        "mixed_task": mixed_task,
        "leaderboard": leaderboard,
        "by_dataset": by_dataset,
        "recommended": winner,
        "reason": reason,
        "metric_keys": metric_keys,
        "param_keys": param_keys,
        "constraints": experiment.get("constraints") or [],
    }
