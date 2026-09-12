"""REST API the workbench UI talks to."""

from __future__ import annotations

import json
import mimetypes
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from workbench import __version__
from workbench.config import CompareConfig, ModelSpec
from workbench.datasets import load_tabular
from workbench.demo import DEMO_SPEC, run_demo
from workbench.ids import new_id
from workbench.models import catalog_for_ui
from workbench.paths import get_store, workbench_home
from workbench.ranking import compare_experiment
from workbench.registry import promote as promote_model
from workbench.store import Store
from workbench.train import start_experiment_from_spec

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class ExperimentCreate(BaseModel):
    name: str
    description: str = ""
    datasets: list[str] = Field(default_factory=lambda: ["iris"])
    models: list[dict[str, Any]] = Field(default_factory=list)
    primary_metric: str | None = None
    maximize: bool | None = True
    test_size: float = 0.2
    random_state: int = 42
    register_model: bool = True
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    start: bool = True
    background: bool = True


class PromoteBody(BaseModel):
    run_id: str | None = None
    version_id: str | None = None
    experiment_id: str | None = None
    model_name: str | None = None
    stage: str = "production"
    note: str
    actor: str = "local"


class DatasetRegister(BaseModel):
    path: str
    name: str
    target: str
    task: str
    description: str = ""


def create_app(store: Store | None = None) -> FastAPI:
    app = FastAPI(title="Workbench", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def current_store() -> Store:
        return store or get_store()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "home": str(workbench_home())}

    @app.get("/api/bootstrap")
    def bootstrap() -> dict[str, Any]:
        db = current_store()
        return {
            "version": __version__,
            "home": str(db.home),
            "stats": db.stats(),
            "catalog": {
                "datasets": db.list_datasets(),
                "models": catalog_for_ui(),
            },
        }

    @app.get("/api/experiments")
    def list_experiments() -> list[dict[str, Any]]:
        db = current_store()
        items = []
        for experiment in db.list_experiments():
            items.append(_experiment_summary(db, experiment))
        return items

    @app.post("/api/experiments")
    def create_experiment(body: ExperimentCreate) -> dict[str, Any]:
        db = current_store()
        if not body.datasets:
            raise HTTPException(400, "Pick at least one dataset")
        models = body.models or [{"name": "random_forest"}]
        spec = body.model_dump()
        spec["experiment"] = body.name
        spec["models"] = models
        if body.start:
            experiment = start_experiment_from_spec(db, spec, background=body.background)
        else:
            experiment = db.create_experiment(
                name=body.name,
                description=body.description,
                primary_metric=body.primary_metric,
                maximize=True if body.maximize is None else body.maximize,
                constraints=body.constraints,
                spec=spec,
                status="draft",
            )
        return _experiment_payload(db, experiment["id"])

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        return _experiment_payload(current_store(), experiment_id)

    @app.get("/api/experiments/{experiment_id}/compare")
    def compare(experiment_id: str, dataset: str | None = Query(default=None)) -> dict[str, Any]:
        db = current_store()
        try:
            experiment = db.get_experiment(experiment_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        runs = db.list_runs(experiment_id)
        payload = compare_experiment(experiment, runs, dataset=dataset)
        payload["experiment"] = _experiment_summary(db, experiment)
        return payload

    @app.post("/api/experiments/{experiment_id}/start")
    def start_existing(experiment_id: str) -> dict[str, Any]:
        db = current_store()
        try:
            experiment = db.get_experiment(experiment_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        spec = experiment.get("spec") or {}
        spec.setdefault("experiment", experiment["name"])
        spec.setdefault("name", experiment["name"])
        db.update_experiment(experiment_id, status="running", error=None)
        cfg = CompareConfig(
            experiment=experiment["name"],
            description=experiment.get("description") or "",
            datasets=list(spec.get("datasets") or ["iris"]),
            models=[ModelSpec(name=item["name"], params=item.get("params") or {}) for item in spec.get("models") or []],
            test_size=float(spec.get("test_size") or 0.2),
            random_state=int(spec.get("random_state") or 42),
            register_model=bool(spec.get("register_model", True)),
            primary_metric=experiment.get("primary_metric"),
            maximize=experiment.get("maximize"),
        )
        from workbench.train import _spawn

        _spawn(db, experiment_id, cfg)
        return _experiment_payload(db, experiment_id)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        db = current_store()
        try:
            run = db.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        run["registry"] = db.versions_for_run(run_id)
        try:
            run["experiment"] = db.get_experiment(run["experiment_id"])
        except KeyError:
            run["experiment"] = None
        return run

    @app.get("/api/artifacts/{artifact_id}")
    def get_artifact(artifact_id: str):
        db = current_store()
        try:
            artifact = db.get_artifact(artifact_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        path = Path(artifact["path"])
        if not path.exists():
            raise HTTPException(404, "Artifact file is missing")
        if artifact.get("mime") == "application/json" or path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("mime", "").startswith("text/") or path.suffix == ".txt":
            return {"text": path.read_text(encoding="utf-8"), **artifact}
        mime = artifact.get("mime") or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=mime, filename=artifact["name"])

    @app.get("/api/datasets")
    def list_datasets() -> list[dict[str, Any]]:
        return current_store().list_datasets()

    @app.post("/api/datasets")
    def register_dataset(body: DatasetRegister) -> dict[str, Any]:
        db = current_store()
        path = Path(body.path).expanduser()
        if not path.exists():
            raise HTTPException(400, f"File not found: {path}")
        try:
            bundle = load_tabular(path, name=body.name, target=body.target, task=body.task, description=body.description)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        dest = db.dataset_root / f"{new_id('file')}_{path.name}"
        shutil.copy2(path, dest)
        slug = body.name.strip().lower().replace(" ", "_")
        try:
            return db.register_dataset(
                slug=slug,
                name=body.name,
                source="file",
                task=bundle.task,
                n_rows=int(bundle.frame.shape[0]),
                n_features=len(bundle.features),
                target=bundle.target,
                feature_names=bundle.features,
                target_names=bundle.target_names,
                description=bundle.description,
                path=str(dest),
                dataset_id=new_id("ds"),
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Could not register dataset: {exc}") from exc

    @app.post("/api/datasets/upload")
    async def upload_dataset(
        file: UploadFile = File(...),
        name: str = Form(...),
        target: str = Form(...),
        task: str = Form(...),
        description: str = Form(""),
    ) -> dict[str, Any]:
        db = current_store()
        dest = db.dataset_root / f"{new_id('file')}_{file.filename or 'data.csv'}"
        dest.write_bytes(await file.read())
        return register_dataset(
            DatasetRegister(path=str(dest), name=name, target=target, task=task, description=description)
        )

    @app.get("/api/registry")
    def registry() -> list[dict[str, Any]]:
        return current_store().list_registry()

    @app.post("/api/promote")
    def promote(body: PromoteBody) -> dict[str, Any]:
        try:
            return promote_model(
                current_store(),
                run_id=body.run_id,
                version_id=body.version_id,
                experiment_id=body.experiment_id,
                model_name=body.model_name,
                stage=body.stage,
                note=body.note,
                actor=body.actor,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/demo")
    def demo(background: bool = True) -> dict[str, Any]:
        db = current_store()
        experiment = run_demo(db, background=background)
        return _experiment_payload(db, experiment["id"])

    @app.get("/api/demo/spec")
    def demo_spec() -> dict[str, Any]:
        return DEMO_SPEC

    def _experiment_summary(db: Store, experiment: dict[str, Any]) -> dict[str, Any]:
        runs = db.list_runs(experiment["id"])
        succeeded = [run for run in runs if run["status"] == "succeeded"]
        comparison = compare_experiment(experiment, runs) if succeeded else None
        winner = (comparison or {}).get("recommended")
        datasets = sorted(
            {
                (run.get("dataset") or {}).get("slug")
                for run in runs
                if (run.get("dataset") or {}).get("slug")
            }
        )
        current = next((run["name"] for run in runs if run["status"] == "running"), None)
        return {
            **experiment,
            "run_count": len(runs),
            "candidate_count": len(succeeded),
            "datasets": datasets,
            "best_metric": (comparison or {}).get("primary_metric"),
            "best_value": winner.get("primary_value") if winner else None,
            "winner_name": winner.get("name") if winner else None,
            "winner_id": winner.get("id") if winner else None,
            "progress": {
                "done": sum(1 for run in runs if run["status"] in {"succeeded", "failed"}),
                "total": len(runs),
                "current": current,
            },
        }

    def _experiment_payload(db: Store, experiment_id: str) -> dict[str, Any]:
        try:
            experiment = db.get_experiment(experiment_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        summary = _experiment_summary(db, experiment)
        summary["runs"] = db.list_runs(experiment_id)
        summary["compare"] = compare_experiment(experiment, summary["runs"])
        return summary

    if FRONTEND_DIST.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            candidate = FRONTEND_DIST / full_path
            if full_path and candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


app = create_app()
