"""Toy pretrained models for demonstration purposes.

The real project aims to ship fully fledged pretrained networks for common
NLP and CV tasks.  To keep the dependencies light for the educational
version of this repository we include extremely small rule based models that
behave like pretrained components.  They allow examples and unit tests to
exercise the evaluation pipeline without requiring large downloads or
external packages.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


class SimpleTextSentimentModel:
    """A minimal sentiment analyser.

    The model is "pretrained" with a tiny vocabulary that marks any text
    containing the word "good" as positive and everything else as negative.
    It mimics the behaviour of a classifier returning ``1`` for positive and
    ``0`` for negative examples.
    """

    positive_token = "good"

    def predict(self, texts: Sequence[str]) -> List[int]:
        return [1 if self.positive_token in t.lower() else 0 for t in texts]


class SimpleImageBrightnessModel:
    """Classify images based on average pixel brightness.

    Each ``image`` in the input is expected to be an iterable of iterables of
    integers in the range 0-255.  Images with an average brightness above 127
    are considered "bright" (class ``1``) and the rest "dark" (class ``0``).
    """

    threshold = 127

    def predict(self, images: Iterable) -> List[int]:
        preds: List[int] = []
        for img in images:
            if isinstance(img, (int, float)):
                mean = float(img)
            else:
                pixels = [p for row in img for p in row]
                mean = sum(pixels) / float(len(pixels)) if pixels else 0.0
            preds.append(1 if mean > self.threshold else 0)
        return preds


def evaluate_classification(model: object, inputs: Sequence, labels: Sequence[int]) -> float:
    """Evaluate a classification model using accuracy.

    Parameters
    ----------
    model:
        Any object exposing a ``predict`` method.
    inputs, labels:
        Sequences of inputs and corresponding ground truth labels.

    Returns
    -------
    float
        The fraction of correctly classified examples.
    """

    preds = model.predict(inputs)  # type: ignore[attr-defined]
    correct = sum(int(p == t) for p, t in zip(preds, labels))
    return correct / float(len(labels)) if labels else 0.0


__all__ = [
    "SimpleTextSentimentModel",
    "SimpleImageBrightnessModel",
    "evaluate_classification",
]

