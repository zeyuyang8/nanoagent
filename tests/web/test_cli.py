from nanoagent.web import cli


def test_help_does_not_load_a_config(capsys) -> None:
    assert cli.main(["--help"]) == 0
    assert "web_cfg=<config.yaml>" in capsys.readouterr().out
