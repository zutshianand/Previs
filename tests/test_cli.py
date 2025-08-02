import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from previs import cli


def test_preprocess_text():
    assert cli.preprocess_text("HeLLo") == "hello"


def test_cli_eval_text(capsys):
    cli.main(["eval-text", "good,bad", "1,0"])
    out = capsys.readouterr().out.strip()
    assert out.endswith("accuracy: 1.00")


def test_cli_eval_image(capsys):
    cli.main(["eval-image", "0,255", "0,1"])
    out = capsys.readouterr().out.strip()
    assert out.endswith("accuracy: 1.00")

