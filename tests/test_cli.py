from workbench.cli import build_parser, main


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


def test_info_exits_zero(capsys) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "iris" in out
    assert "classification" in out
