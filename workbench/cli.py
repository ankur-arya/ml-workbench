"""Command-line interface for the MLflow sklearn workbench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from workbench.config import (
    DEFAULT_EXPERIMENT,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRACKING_URI,
    TrainConfig,
    config_from_mapping,
    load_yaml,
    merge_config,
    parse_param_overrides,
)
from workbench.datasets import list_datasets, load_dataset
from workbench.models import list_models
from workbench.tracking import resolve_tracking_uri
from workbench.train import run_comparison, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m workbench",
        description="Train scikit-learn models and log runs to a local MLflow store.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help=(
            "MLflow tracking URI (default: $MLFLOW_TRACKING_URI or "
            f"{DEFAULT_TRACKING_URI}). Use http://127.0.0.1:5000 when a "
            "local `mlflow server` is already running."
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train one model on one dataset and log it.")
    train.add_argument("--config", type=Path, help="Optional YAML config (CLI flags override it).")
    train.add_argument("--experiment", default=None, help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT}).")
    train.add_argument("--dataset", default=None, choices=list_datasets(), help="sklearn dataset name.")
    train.add_argument("--model", default=None, help="Model key, e.g. random_forest, logistic_regression, ridge.")
    train.add_argument("--test-size", type=float, default=None, help=f"Holdout fraction (default: {DEFAULT_TEST_SIZE}).")
    train.add_argument("--random-state", type=int, default=None, help=f"RNG seed (default: {DEFAULT_RANDOM_STATE}).")
    train.add_argument("--run-name", default=None, help="Optional MLflow run name.")
    train.add_argument(
        "--param",
        action="append",
        dest="params",
        metavar="KEY=VALUE",
        help="Estimator hyperparameter override (repeatable), e.g. --param n_estimators=200.",
    )
    train.add_argument(
        "--register",
        dest="register_model",
        action="store_true",
        default=None,
        help="Register the logged model (default: on).",
    )
    train.add_argument(
        "--no-register",
        dest="register_model",
        action="store_false",
        help="Skip model registry (model is still logged on the run).",
    )

    compare = sub.add_parser(
        "compare",
        help="Train several models (nested MLflow runs) for one or more datasets.",
    )
    compare.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        choices=list_datasets(),
        help="Dataset to include (repeatable). Default: iris, wine, diabetes.",
    )
    compare.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model key to include (repeatable). Default: all models valid for each dataset task.",
    )
    compare.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    compare.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    compare.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    compare.add_argument(
        "--no-register",
        dest="register_model",
        action="store_false",
        default=True,
        help="Skip model registry for comparison runs.",
    )

    info = sub.add_parser("info", help="List supported datasets and models.")
    info.add_argument("--dataset", choices=list_datasets(), help="Show models for one dataset's task.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    tracking_uri = resolve_tracking_uri(args.tracking_uri)

    if args.command == "info":
        return _cmd_info(args.dataset)

    if args.command == "train":
        cfg = TrainConfig(tracking_uri=tracking_uri)
        if args.config:
            cfg = merge_config(cfg, **config_from_mapping(load_yaml(args.config)).__dict__)
        cfg = merge_config(
            cfg,
            experiment=args.experiment,
            dataset=args.dataset,
            model=args.model,
            test_size=args.test_size,
            random_state=args.random_state,
            run_name=args.run_name,
            register_model=args.register_model,
            tracking_uri=tracking_uri,
            params=parse_param_overrides(args.params),
        )
        run_training(cfg)
        return 0

    if args.command == "compare":
        datasets = args.datasets or ["iris", "wine", "diabetes"]
        run_comparison(
            datasets,
            args.models,
            experiment=args.experiment,
            tracking_uri=tracking_uri,
            test_size=args.test_size,
            random_state=args.random_state,
            register_model=args.register_model,
        )
        return 0

    parser.error(f"Unknown command {args.command}")
    return 2


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
