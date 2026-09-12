"""SQLite store for experiments, runs, datasets, artifacts, and the registry."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workbench.datasets import DATASET_META, snapshot_builtin
from workbench.ids import new_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  primary_metric TEXT,
  maximize INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'draft',
  constraints_json TEXT NOT NULL DEFAULT '[]',
  spec_json TEXT NOT NULL DEFAULT '{}',
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  source TEXT NOT NULL,
  task TEXT NOT NULL,
  n_rows INTEGER,
  n_features INTEGER,
  target TEXT,
  feature_names_json TEXT,
  target_names_json TEXT,
  description TEXT NOT NULL DEFAULT '',
  path TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  experiment_id TEXT NOT NULL,
  dataset_id TEXT,
  name TEXT NOT NULL,
  model_name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  task TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  FOREIGN KEY (experiment_id) REFERENCES experiments(id),
  FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);

CREATE TABLE IF NOT EXISTS params (
  run_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  PRIMARY KEY (run_id, key),
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value REAL NOT NULL,
  PRIMARY KEY (run_id, key),
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  mime TEXT,
  label TEXT,
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS registered_models (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  description TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
  id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  run_id TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT 'candidate',
  created_at TEXT NOT NULL,
  UNIQUE (model_id, version),
  FOREIGN KEY (model_id) REFERENCES registered_models(id),
  FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS promotions (
  id TEXT PRIMARY KEY,
  version_id TEXT NOT NULL,
  from_stage TEXT NOT NULL,
  to_stage TEXT NOT NULL,
  note TEXT NOT NULL,
  actor TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (version_id) REFERENCES model_versions(id)
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_versions_model ON model_versions(model_id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


class Store:
    def __init__(self, home: str | Path):
        self.home = Path(home).expanduser().resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.db_path = self.home / "workbench.db"
        self.artifact_root = self.home / "artifacts"
        self.dataset_root = self.home / "datasets"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init(self) -> None:
        with self._lock, self.connect() as conn:
            conn.executescript(SCHEMA)
        self.seed_builtin_datasets()

    def seed_builtin_datasets(self) -> None:
        with self._lock, self.connect() as conn:
            for slug, meta in DATASET_META.items():
                existing = conn.execute("SELECT id FROM datasets WHERE slug = ?", (slug,)).fetchone()
                if existing:
                    continue
                snap = snapshot_builtin(slug)
                conn.execute(
                    """
                    INSERT INTO datasets (
                      id, slug, name, source, task, n_rows, n_features, target,
                      feature_names_json, target_names_json, description, path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        slug,
                        slug,
                        snap["name"],
                        "builtin",
                        snap["task"],
                        snap["n_rows"],
                        snap["n_features"],
                        snap["target"],
                        json.dumps(snap["feature_names"]),
                        json.dumps(snap["target_names"]),
                        snap["description"],
                        None,
                        utcnow(),
                    ),
                )

    # --- experiments -------------------------------------------------
    def create_experiment(
        self,
        *,
        name: str,
        description: str = "",
        primary_metric: str | None = None,
        maximize: bool = True,
        constraints: list[dict[str, Any]] | None = None,
        spec: dict[str, Any] | None = None,
        status: str = "draft",
    ) -> dict[str, Any]:
        exp_id = new_id("exp")
        now = utcnow()
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                  id, name, description, primary_metric, maximize, status,
                  constraints_json, spec_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp_id,
                    name,
                    description or "",
                    primary_metric,
                    1 if maximize else 0,
                    status,
                    json.dumps(constraints or []),
                    json.dumps(spec or {}),
                    now,
                    now,
                ),
            )
        return self.get_experiment(exp_id)

    def update_experiment(self, exp_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {
            "name",
            "description",
            "primary_metric",
            "maximize",
            "status",
            "error",
            "constraints_json",
            "spec_json",
        }
        payload = {key: value for key, value in fields.items() if key in allowed}
        if "maximize" in payload and not isinstance(payload["maximize"], str):
            payload["maximize"] = 1 if payload["maximize"] else 0
        if not payload:
            return self.get_experiment(exp_id)
        payload["updated_at"] = utcnow()
        assignments = ", ".join(f"{key} = ?" for key in payload)
        values = list(payload.values()) + [exp_id]
        with self._lock, self.connect() as conn:
            conn.execute(f"UPDATE experiments SET {assignments} WHERE id = ?", values)
        return self.get_experiment(exp_id)

    def get_experiment(self, exp_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown experiment {exp_id}")
        return self._hydrate_experiment(dict(row))

    def list_experiments(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM experiments ORDER BY updated_at DESC").fetchall()
        return [self._hydrate_experiment(dict(row)) for row in rows]

    def _hydrate_experiment(self, row: dict[str, Any]) -> dict[str, Any]:
        row["maximize"] = bool(row["maximize"])
        row["constraints"] = json.loads(row.pop("constraints_json") or "[]")
        row["spec"] = json.loads(row.pop("spec_json") or "{}")
        return row

    # --- datasets ----------------------------------------------------
    def list_datasets(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY source, name").fetchall()
            used = {
                item["dataset_id"]: item["n"]
                for item in conn.execute(
                    "SELECT dataset_id, COUNT(*) AS n FROM runs WHERE dataset_id IS NOT NULL GROUP BY dataset_id"
                )
            }
        out = []
        for row in rows:
            item = self._hydrate_dataset(dict(row))
            item["used_in_runs"] = used.get(item["id"], 0)
            out.append(item)
        return out

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM datasets WHERE id = ? OR slug = ?",
                (dataset_id, dataset_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown dataset {dataset_id}")
        return self._hydrate_dataset(dict(row))

    def _hydrate_dataset(self, row: dict[str, Any]) -> dict[str, Any]:
        row["feature_names"] = json.loads(row.pop("feature_names_json") or "[]")
        row["target_names"] = json.loads(row.pop("target_names_json") or "null")
        return row

    def register_dataset(
        self,
        *,
        slug: str,
        name: str,
        source: str,
        task: str,
        n_rows: int | None,
        n_features: int | None,
        target: str | None,
        feature_names: list[str] | None = None,
        target_names: list[str] | None = None,
        description: str = "",
        path: str | None = None,
        dataset_id: str | None = None,
    ) -> dict[str, Any]:
        ds_id = dataset_id or slug
        now = utcnow()
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets (
                  id, slug, name, source, task, n_rows, n_features, target,
                  feature_names_json, target_names_json, description, path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ds_id,
                    slug,
                    name,
                    source,
                    task,
                    n_rows,
                    n_features,
                    target,
                    json.dumps(feature_names or []),
                    json.dumps(target_names),
                    description,
                    path,
                    now,
                ),
            )
        return self.get_dataset(ds_id)

    # --- runs --------------------------------------------------------
    def create_run(
        self,
        *,
        experiment_id: str,
        name: str,
        model_name: str,
        dataset_id: str | None,
        task: str | None = None,
        status: str = "pending",
    ) -> dict[str, Any]:
        run_id = new_id("run")
        now = utcnow()
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                  id, experiment_id, dataset_id, name, model_name, status, task, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, experiment_id, dataset_id, name, model_name, status, task, now),
            )
        return self.get_run(run_id)

    def update_run(self, run_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {"status", "error", "started_at", "finished_at", "task", "name"}
        payload = {key: value for key, value in fields.items() if key in allowed}
        if not payload:
            return self.get_run(run_id)
        assignments = ", ".join(f"{key} = ?" for key in payload)
        values = list(payload.values()) + [run_id]
        with self._lock, self.connect() as conn:
            conn.execute(f"UPDATE runs SET {assignments} WHERE id = ?", values)
            conn.execute("UPDATE experiments SET updated_at = ? WHERE id = (SELECT experiment_id FROM runs WHERE id = ?)", (utcnow(), run_id))
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown run {run_id}")
            params = {
                item["key"]: _maybe_json(item["value"])
                for item in conn.execute("SELECT key, value FROM params WHERE run_id = ?", (run_id,))
            }
            metrics = {
                item["key"]: item["value"]
                for item in conn.execute("SELECT key, value FROM metrics WHERE run_id = ?", (run_id,))
            }
            artifacts = [dict(item) for item in conn.execute("SELECT * FROM artifacts WHERE run_id = ?", (run_id,))]
            dataset = None
            if row["dataset_id"]:
                ds = conn.execute("SELECT * FROM datasets WHERE id = ?", (row["dataset_id"],)).fetchone()
                dataset = self._hydrate_dataset(dict(ds)) if ds else None
        payload = dict(row)
        payload["params"] = params
        payload["metrics"] = metrics
        payload["artifacts"] = artifacts
        payload["dataset"] = dataset
        return payload

    def list_runs(self, experiment_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT id FROM runs"
        args: tuple[Any, ...] = ()
        if experiment_id:
            query += " WHERE experiment_id = ?"
            args = (experiment_id,)
        query += " ORDER BY created_at ASC"
        with self.connect() as conn:
            ids = [row["id"] for row in conn.execute(query, args)]
        return [self.get_run(run_id) for run_id in ids]

    def set_params(self, run_id: str, params: dict[str, Any]) -> None:
        with self._lock, self.connect() as conn:
            for key, value in params.items():
                conn.execute(
                    "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
                    (run_id, key, _stringify(value)),
                )

    def set_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        with self._lock, self.connect() as conn:
            for key, value in metrics.items():
                conn.execute(
                    "INSERT OR REPLACE INTO metrics (run_id, key, value) VALUES (?, ?, ?)",
                    (run_id, key, float(value)),
                )

    def add_artifact(
        self,
        run_id: str,
        *,
        name: str,
        kind: str,
        path: str,
        mime: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        art_id = new_id("art")
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (id, run_id, name, kind, path, mime, label)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (art_id, run_id, name, kind, path, mime, label),
            )
        return {"id": art_id, "run_id": run_id, "name": name, "kind": kind, "path": path, "mime": mime, "label": label}

    def get_artifact(self, artifact_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown artifact {artifact_id}")
        return dict(row)

    def artifact_dir(self, run_id: str) -> Path:
        path = self.artifact_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --- registry ----------------------------------------------------
    def get_or_create_model(self, name: str, description: str = "") -> dict[str, Any]:
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT * FROM registered_models WHERE name = ?", (name,)).fetchone()
            if row:
                return dict(row)
            model_id = new_id("mdl")
            now = utcnow()
            conn.execute(
                """
                INSERT INTO registered_models (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (model_id, name, description, now, now),
            )
        return self.get_registered_model(name)

    def get_registered_model(self, name_or_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM registered_models WHERE name = ? OR id = ?",
                (name_or_id, name_or_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown registered model {name_or_id}")
        return dict(row)

    def add_model_version(self, model_name: str, run_id: str, stage: str = "candidate") -> dict[str, Any]:
        model = self.get_or_create_model(model_name)
        with self._lock, self.connect() as conn:
            current = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM model_versions WHERE model_id = ?",
                (model["id"],),
            ).fetchone()
            version = int(current["v"]) + 1
            version_id = new_id("ver")
            now = utcnow()
            conn.execute(
                """
                INSERT INTO model_versions (id, model_id, version, run_id, stage, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (version_id, model["id"], version, run_id, stage, now),
            )
            conn.execute(
                "UPDATE registered_models SET updated_at = ? WHERE id = ?",
                (now, model["id"]),
            )
        return self.get_version(version_id)

    def get_version(self, version_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM model_versions WHERE id = ?", (version_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown model version {version_id}")
        return dict(row)

    def versions_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT v.*, m.name AS model_name
                FROM model_versions v
                JOIN registered_models m ON m.id = v.model_id
                WHERE v.run_id = ?
                ORDER BY v.version DESC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_version_stage(self, version_id: str, stage: str) -> dict[str, Any]:
        with self._lock, self.connect() as conn:
            conn.execute("UPDATE model_versions SET stage = ? WHERE id = ?", (stage, version_id))
            conn.execute(
                """
                UPDATE registered_models SET updated_at = ?
                WHERE id = (SELECT model_id FROM model_versions WHERE id = ?)
                """,
                (utcnow(), version_id),
            )
        return self.get_version(version_id)

    def archive_stage(self, model_id: str, stage: str, except_version: str | None = None) -> list[str]:
        query = "SELECT id FROM model_versions WHERE model_id = ? AND stage = ?"
        args: list[Any] = [model_id, stage]
        if except_version:
            query += " AND id != ?"
            args.append(except_version)
        with self._lock, self.connect() as conn:
            ids = [row["id"] for row in conn.execute(query, args)]
            for version_id in ids:
                conn.execute("UPDATE model_versions SET stage = ? WHERE id = ?", ("archived", version_id))
        return ids

    def add_promotion(
        self,
        *,
        version_id: str,
        from_stage: str,
        to_stage: str,
        note: str,
        actor: str,
    ) -> dict[str, Any]:
        promo_id = new_id("prm")
        now = utcnow()
        with self._lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO promotions (id, version_id, from_stage, to_stage, note, actor, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (promo_id, version_id, from_stage, to_stage, note, actor, now),
            )
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM promotions WHERE id = ?", (promo_id,)).fetchone()
        return dict(row)

    def list_registry(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            models = [dict(row) for row in conn.execute("SELECT * FROM registered_models ORDER BY updated_at DESC")]
            out = []
            for model in models:
                versions = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT v.*, r.name AS run_name, r.model_name AS estimator, r.experiment_id
                        FROM model_versions v
                        JOIN runs r ON r.id = v.run_id
                        WHERE v.model_id = ?
                        ORDER BY v.version DESC
                        """,
                        (model["id"],),
                    )
                ]
                for version in versions:
                    version["promotions"] = [
                        dict(row)
                        for row in conn.execute(
                            "SELECT * FROM promotions WHERE version_id = ? ORDER BY created_at DESC",
                            (version["id"],),
                        )
                    ]
                production = next((item for item in versions if item["stage"] == "production"), None)
                model["versions"] = versions
                model["production"] = production
                model["alias"] = "champion" if production else None
                out.append(model)
        return out

    def stats(self) -> dict[str, int]:
        with self.connect() as conn:
            experiments = conn.execute("SELECT COUNT(*) AS n FROM experiments").fetchone()["n"]
            runs = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
            production = conn.execute(
                "SELECT COUNT(*) AS n FROM model_versions WHERE stage = 'production'"
            ).fetchone()["n"]
            datasets = conn.execute("SELECT COUNT(*) AS n FROM datasets").fetchone()["n"]
        return {
            "experiments": experiments,
            "runs": runs,
            "production_models": production,
            "datasets": datasets,
        }


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _maybe_json(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    if value == "null":
        return None
    try:
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        return float(value)
    except ValueError:
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
