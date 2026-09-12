"""Command-line interface for the ClearML sklearn workbench."""

from __future__ import annotations

import argparse
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
from workbench.models import list_models
from workbench.promote import promote_model
from workbench.train import run_comparison, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m workbench",
        description=(
            "Train scikit-learn models and log tasks to ClearML "
            "(local Server / WebApp or hosted app.clear.ml)."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Log to a local ClearML offline session (no server). Import later with Task.import_offline_session.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train one model on one dataset and log it.")
    train.add_argument("--config", type=Path, help="Optional YAML config (CLI flags override it).")
    train.add_argument(
        "--experiment",
        default=None,
        help=f"ClearML project name (default: {DEFAULT_EXPERIMENT}).",
    )
    train.add_argument("--project", default=None, help="Alias for --experiment (ClearML project).")
    train.add_argument("--dataset", default=None, choices=list_datasets(), help="sklearn dataset name.")
    train.add_argument("--model", default=None, help="Model key, e.g. random_forest, logistic_regression, ridge.")
    train.add_argument("--test-size", type=float, default=None, help=f"Holdout fraction (default: {DEFAULT_TEST_SIZE}).")
    train.add_argument("--random-state", type=int, default=None, help=f"RNG seed (default: {DEFAULT_RANDOM_STATE}).")
    train.add_argument("--run-name", default=None, help="Optional ClearML task name.")
    train.add_argument(
        "--param",
        action="append",
        dest="params",
        metavar="KEY=VALUE",
        help="Estimator hyperparameter override (repeatable), e.g. --param n_estimators=200.",
    )
    train.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="ClearML task tag (repeatable).",
    )
    train.add_argument(
        "--register",
        dest="register_model",
        action="store_true",
        default=None,
        help="Register an OutputModel in the ClearML catalog (default: on).",
    )
    train.add_argument(
        "--no-register",
        dest="register_model",
        action="store_false",
        help="Skip OutputModel registration (metrics/plots are still logged).",
    )
    train.add_argument(
        "--register-dataset",
        dest="register_dataset",
        action="store_true",
        default=None,
        help="Also upload the sklearn table as a ClearML Dataset.",
    )

    compare = sub.add_parser(
        "compare",
        help="Train several models/datasets as sibling ClearML tasks in one project.",
    )
    compare.add_argument("--config", type=Path, help="Optional YAML with datasets/models lists.")
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
    compare.add_argument("--experiment", default=None)
    compare.add_argument("--project", default=None, help="Alias for --experiment.")
    compare.add_argument("--test-size", type=float, default=None)
    compare.add_argument("--random-state", type=int, default=None)
    compare.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Extra ClearML tags applied to every comparison task.",
    )
    compare.add_argument(
        "--no-register",
        dest="register_model",
        action="store_false",
        default=None,
        help="Skip OutputModel registration for comparison runs.",
    )

    promote = sub.add_parser(
        "promote",
        help="Tag + publish a ClearML model (registry promote toward production).",
    )
    promote.add_argument("--model-id", help="ClearML model id to publish.")
    promote.add_argument("--task-id", help="Use the first output model of this task.")
    promote.add_argument(
        "--experiment",
        "--project",
        dest="project",
        default=None,
        help="Pick the best output model in this ClearML project by --metric.",
    )
    promote.add_argument(
        "--metric",
        default="test_accuracy",
        help="Scalar used when selecting the best model in a project (default: test_accuracy).",
    )
    promote.add_argument(
        "--minimize",
        action="store_true",
        help="Treat lower metric values as better (e.g. test_rmse).",
    )
    promote.add_argument(
        "--tag",
        action="append",
        dest="tags",
        help="Tag to apply (repeatable). Default: production.",
    )
    promote.add_argument(
        "--publish",
        dest="publish",
        action="store_true",
        default=True,
        help="Publish the model (default: on).",
    )
    promote.add_argument(
        "--no-publish",
        dest="publish",
        action="store_false",
        help="Only apply tags; do not publish.",
    )
    promote.add_argument(
        "--compare-tag",
        default=None,
        help="Limit auto-select to tasks that have this ClearML tag.",
    )

    info = sub.add_parser("info", help="List supported datasets and models.")
    info.add_argument("--dataset", choices=list_datasets(), help="Show models for one dataset's task.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "info":
        return _cmd_info(args.dataset)

    if args.command == "train":
        cfg = TrainConfig(offline=args.offline)
        if args.config:
            cfg = merge_config(cfg, **config_from_mapping(load_yaml(args.config)).__dict__)
        experiment = args.project or args.experiment
        cfg = merge_config(
            cfg,
            experiment=experiment,
            dataset=args.dataset,
            model=args.model,
            test_size=args.test_size,
            random_state=args.random_state,
            run_name=args.run_name,
            register_model=args.register_model,
            register_dataset=args.register_dataset,
            offline=True if args.offline else None,
            params=parse_param_overrides(args.params),
            tags=args.tags,
        )
        run_training(cfg)
        return 0

    if args.command == "compare":
        compare_cfg = CompareConfig(offline=args.offline)
        if args.config:
            loaded = compare_config_from_mapping(load_yaml(args.config))
            compare_cfg = loaded
            if args.offline:
                compare_cfg.offline = True
        experiment = args.project or args.experiment or compare_cfg.experiment
        datasets = args.datasets or compare_cfg.datasets
        models = args.models if args.models is not None else compare_cfg.models
        register = compare_cfg.register_model if args.register_model is None else args.register_model
        test_size = compare_cfg.test_size if args.test_size is None else args.test_size
        random_state = compare_cfg.random_state if args.random_state is None else args.random_state
        tags = list(compare_cfg.tags)
        if args.tags:
            tags.extend(args.tags)
        run_comparison(
            datasets,
            models,
            experiment=experiment,
            test_size=test_size,
            random_state=random_state,
            register_model=register,
            register_dataset=compare_cfg.register_dataset,
            offline=args.offline or compare_cfg.offline,
            tags=tags,
        )
        return 0

    if args.command == "promote":
        promote_model(
            model_id=args.model_id,
            task_id=args.task_id,
            project=args.project,
            metric=args.metric,
            higher_is_better=False if args.minimize else None,
            tags=args.tags or ["production"],
            publish=args.publish,
            compare_tag=args.compare_tag,
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
