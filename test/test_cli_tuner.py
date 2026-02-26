from vhs_pipeline.cli import build_parser


def test_cli_has_tuner_command_with_host_and_port() -> None:
    parser = build_parser()
    args = parser.parse_args(["tuner", "--host", "127.0.0.1", "--port", "9001"])
    assert args.group == "tuner"
    assert args.host == "127.0.0.1"
    assert args.port == 9001
