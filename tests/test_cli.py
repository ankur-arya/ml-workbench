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


def test_parser_promote_and_ui() -> None:
    parser = build_parser()
    promote = parser.parse_args(["promote", "--experiment", "exp_1", "--note", "Ship it"])
    assert promote.command == "promote"
    assert promote.note == "Ship it"
    ui = parser.parse_args(["ui", "--port", "9000"])
    assert ui.port == 9000


def test_info_exits_zero(capsys) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "iris" in out
    assert "classification" in out
