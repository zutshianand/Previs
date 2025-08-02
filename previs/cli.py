"""Command line utilities for :mod:`previs`.

The CLI currently exposes lightweight helpers for running preprocessing
steps and evaluating the toy pretrained models contained in
``previs.models.pretrained``.  It can be invoked either via

``python -m previs.cli``

or after installation through the ``previs`` console script.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Sequence

from .models import pretrained


def preprocess_text(text: str) -> str:
    """A trivial preprocessing pipeline used for demonstration.

    It simply lowercases the provided text.
    """

    return text.lower()


def _cmd_preprocess(args: argparse.Namespace) -> None:
    print(preprocess_text(args.text))


def _cmd_eval_text(args: argparse.Namespace) -> None:
    model = pretrained.SimpleTextSentimentModel()
    texts: Sequence[str] = args.texts.split(",")
    labels = [int(x) for x in args.labels.split(",")]
    acc = pretrained.evaluate_classification(model, texts, labels)
    print(f"accuracy: {acc:.2f}")


def _cmd_eval_image(args: argparse.Namespace) -> None:
    model = pretrained.SimpleImageBrightnessModel()
    images: Iterable[float] = (float(x) for x in args.images.split(","))
    labels = [int(x) for x in args.labels.split(",")]
    acc = pretrained.evaluate_classification(model, images, labels)
    print(f"accuracy: {acc:.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="previs", description="Previs utilities")
    sub = parser.add_subparsers(dest="command")

    p_pre = sub.add_parser("preprocess", help="Lowercase input text")
    p_pre.add_argument("text", help="Text to preprocess")
    p_pre.set_defaults(func=_cmd_preprocess)

    p_eval_t = sub.add_parser("eval-text", help="Evaluate the toy text model")
    p_eval_t.add_argument("texts", help="Comma separated texts")
    p_eval_t.add_argument("labels", help="Comma separated integer labels")
    p_eval_t.set_defaults(func=_cmd_eval_text)

    p_eval_i = sub.add_parser("eval-image", help="Evaluate the toy image model")
    p_eval_i.add_argument("images", help="Comma separated average brightness values")
    p_eval_i.add_argument("labels", help="Comma separated integer labels")
    p_eval_i.set_defaults(func=_cmd_eval_image)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

