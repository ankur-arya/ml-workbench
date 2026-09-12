from workbench.cli import build_parser, main
from workbench.config import compare_config_from_mapping, load_yaml


def test_parser_train_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset",
            "wine",
            "--model",
            "random_forest",
            "--param",
            "n_estimators=10",
        ]
    )
    assert args.command == "train"
    assert args.dataset == "wine"
    assert args.params == ["n_estimators=10"]


def test_parser_promote_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "promote",
            "--model-id",
            "abc123",
            "--tag",
            "production",
            "--tag",
            "champion",
            "--no-publish",
        ]
    )
    assert args.command == "promote"
    assert args.model_id == "abc123"
    assert args.tags == ["production", "champion"]
    assert args.publish is False


def test_compare_yaml_roundtrip(tmp_path) -> None:
    path = tmp_path / "compare.yaml"
    path.write_text(
        "experiment: demo\ndatasets:\n  - iris\nmodels:\n  - random_forest\n",
        encoding="utf-8",
    )
    cfg = compare_config_from_mapping(load_yaml(path))
    assert cfg.experiment == "demo"
    assert cfg.datasets == ["iris"]
    assert cfg.models == ["random_forest"]


def test_info_exits_zero(capsys) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "iris" in out
    assert "classification" in out


def test_train_cli_with_memory_tracker(capsys) -> None:
    assert (
        main(
            [
                "train",
                "--dataset",
                "iris",
                "--model",
                "logistic_regression",
                "--param",
                "max_iter=200",
                "--no-register",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "dataset=iris" in out
    assert "test_accuracy=" in out
