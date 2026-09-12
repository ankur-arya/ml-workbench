"""Command-line interface: train, compare, promote, ui."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from workbench.config import (
    DEFAULT_EXPERIMENT,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    CompareConfig,
    TrainConfig,
    compare_config_from_mapping,
    config_from_mapping,
    load_yaml,
    merge_config,
    parse_param_overrides,
)
from workbench.datasets import list_datasets, load_dataset
from workbench.demo import run_demo
from workbench.models import list_models
from workbench.paths import get_store, reset_store, workbench_home
from workbench.ranking import compare_experiment
from workbench.registry import promote
from workbench.train import run_comparison, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workbench",
        description="Local-first classical ML workbench — evaluate, compare, promote. No Docker.",
    )
    parser.add_argument(
        "--home",
        default=None,
        help="Data directory (default: $WORKBENCH_HOME or ./.workbench).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train one model on one dataset.")
    train.add_argument("--config", type=Path, help="Optional YAML config (CLI flags override it).")
    train.add_argument("--experiment", default=None, help=f"Experiment name (default: {DEFAULT_EXPERIMENT}).")
    train.add_argument("--dataset", default=None, help="Dataset slug (builtin or registered).")
    train.add_argument("--model", default=None, help="Model key, e.g. random_forest.")
    train.add_argument("--test-size", type=float, default=None)
    train.add_argument("--random-state", type=int, default=None)
    train.add_argument("--run-name", default=None)
    train.add_argument("--description", default=None)
    train.add_argument("--param", action="append", dest="params", metavar="KEY=VALUE")
    train.add_argument("--register", dest="register_model", action="store_true", default=None)
    train.add_argument("--no-register", dest="register_model", action="store_false")

    compare = sub.add_parser("compare", help="Train several models and/or datasets as one experiment.")
    compare.add_argument("--config", type=Path, help="YAML with datasets/models lists.")
    compare.add_argument("--dataset", action="append", dest="datasets")
    compare.add_argument("--model", action="append", dest="models")
    compare.add_argument("--experiment", default=None)
    compare.add_argument("--description", default=None)
    compare.add_argument("--primary-metric", default=None)
    compare.add_argument("--minimize", action="store_true", help="Lower primary metric is better.")
    compare.add_argument("--test-size", type=float, default=None)
    compare.add_argument("--random-state", type=int, default=None)
    compare.add_argument("--no-register", dest="register_model", action="store_false", default=None)

    promote_p = sub.add_parser("promote", help="Promote a run or the experiment winner to a registry stage.")
    promote_p.add_argument("--run", dest="run_id", help="Run id to promote.")
    promote_p.add_argument("--experiment", dest="experiment_id", help="Promote the recommended winner.")
    promote_p.add_argument("--version", dest="version_id", help="Existing model version id.")
    promote_p.add_argument("--stage", default="production", help="candidate | staging | production | archived")
    promote_p.add_argument("--note", required=True, help="Required reason for the promotion.")
    promote_p.add_argument("--actor", default=os.environ.get("USER", "local"))
    promote_p.add_argument("--model-name", default=None)

    ui = sub.add_parser("ui", help="Start the local API + UI (no Docker).")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8000)
    ui.add_argument("--dev", action="store_true", help="Also print the Vite dev-server hint.")
    ui.add_argument("--reload", action="store_true")
    ui.add_argument("--skip-build", action="store_true", help="Do not auto-build the frontend.")

    demo = sub.add_parser("demo", help="Run the built-in iris bakeoff experiment.")

    info = sub.add_parser("info", help="List supported datasets and models.")
    info.add_argument("--dataset", help="Show models for one dataset's task.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.home:
        os.environ["WORKBENCH_HOME"] = str(Path(args.home).expanduser().resolve())
        reset_store()

    if args.command == "info":
        return _cmd_info(args.dataset)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "promote":
        return _cmd_promote(args)
    if args.command == "demo":
        return _cmd_demo()
    if args.command == "ui":
        return _cmd_ui(args)
    parser.error(f"Unknown command {args.command}")
    return 2


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = TrainConfig()
    if args.config:
        cfg = merge_config(cfg, **config_from_mapping(load_yaml(args.config)).__dict__)
    cfg = merge_config(
        cfg,
        experiment=args.experiment,
        description=args.description,
        dataset=args.dataset,
        model=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
        run_name=args.run_name,
        register_model=args.register_model,
        params=parse_param_overrides(args.params),
    )
    run = run_training(cfg)
    print(f"Experiment {run['experiment_id']}  run {run['id']}")
    return 0 if run["status"] == "succeeded" else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    data: dict = {}
    if args.config:
        data = load_yaml(args.config)
    cfg = compare_config_from_mapping(data) if data else CompareConfig()
    if args.datasets:
        cfg.datasets = args.datasets
    if args.models:
        from workbench.config import ModelSpec

        cfg.models = [ModelSpec(name=name) for name in args.models]
    if args.experiment:
        cfg.experiment = args.experiment
    if args.description:
        cfg.description = args.description
    if args.primary_metric:
        cfg.primary_metric = args.primary_metric
    if args.minimize:
        cfg.maximize = False
    if args.test_size is not None:
        cfg.test_size = args.test_size
    if args.random_state is not None:
        cfg.random_state = args.random_state
    if args.register_model is not None:
        cfg.register_model = args.register_model
    if not args.config and not args.datasets:
        cfg.datasets = ["iris"]
        if cfg.models is None:
            from workbench.config import ModelSpec

            cfg.models = [
                ModelSpec(name="logistic_regression"),
                ModelSpec(name="random_forest"),
                ModelSpec(name="gradient_boosting"),
            ]

    experiment = run_comparison(cfg)
    store = get_store()
    scorecard = compare_experiment(store.get_experiment(experiment["id"]), store.list_runs(experiment["id"]))
    print(f"\nExperiment {experiment['id']}  {experiment['name']}")
    print(f"Primary metric: {scorecard['primary_metric']}  ({'↑' if scorecard['maximize'] else '↓'} better)")
    print("")
    print(f"{'#':>3}  {'model':<22} {'dataset':<16} {scorecard['primary_metric']}")
    for row in scorecard["leaderboard"]:
        if row.get("rank") is None:
            continue
        slug = (row.get("dataset") or {}).get("slug", "")
        value = row.get("primary_value")
        mark = " ← winner" if row.get("is_recommended") else ""
        print(f"{row['rank']:>3}  {row['model_name']:<22} {slug:<16} {value:.4f}{mark}")
    winner = scorecard.get("recommended")
    if winner:
        print(f"\nRecommended: {winner['name']}  ({winner['id']})")
        print(f"Promote: python -m workbench promote --experiment {experiment['id']} --note \"…\"")
    return 0 if experiment["status"] == "completed" else 1


def _cmd_promote(args: argparse.Namespace) -> int:
    store = get_store()
    result = promote(
        store,
        run_id=args.run_id,
        version_id=args.version_id,
        experiment_id=args.experiment_id,
        model_name=args.model_name,
        stage=args.stage,
        note=args.note,
        actor=args.actor,
    )
    version = result["version"]
    model = result["model"]
    print(
        f"Promoted {model['name']} v{version['version']} → {version['stage']}"
        + (f" ({result['alias']})" if result.get("alias") else "")
    )
    print(f"  by {result['promotion']['actor']}: {result['promotion']['note']}")
    print(f"  source run {result['run']['id']}")
    return 0


def _cmd_demo() -> int:
    experiment = run_demo(get_store(), background=False)
    print(f"Demo experiment {experiment['id']}  status={experiment['status']}")
    return 0 if experiment["status"] == "completed" else 1


def _cmd_ui(args: argparse.Namespace) -> int:
    from workbench.server import serve

    print(f"Workbench data: {workbench_home()}")
    print(f"Open http://{args.host}:{args.port}")
    serve(host=args.host, port=args.port, reload=args.reload, skip_build=args.skip_build)
    return 0


def _cmd_info(dataset_name: str | None) -> int:
    if dataset_name:
        bundle = load_dataset(dataset_name)
        print(f"{bundle.name}  task={bundle.task}  {bundle.description}")
        print("models:", ", ".join(list_models(bundle.task)))
        return 0
    print("datasets:")
    for name in list_datasets():
        bundle = load_dataset(name)
        print(f"  {bundle.name:16} {bundle.task:16} {bundle.description}")
    print("\nclassification models:", ", ".join(list_models("classification")))
    print("regression models:    ", ", ".join(list_models("regression")))
    print(f"\nlocal store: {workbench_home()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
