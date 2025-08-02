import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from previs.models import pretrained


def test_text_model_accuracy():
    model = pretrained.SimpleTextSentimentModel()
    texts = ["good", "bad"]
    labels = [1, 0]
    acc = pretrained.evaluate_classification(model, texts, labels)
    assert acc == 1.0


def test_image_model_accuracy_with_numbers():
    model = pretrained.SimpleImageBrightnessModel()
    images = [10, 200]
    labels = [0, 1]
    acc = pretrained.evaluate_classification(model, images, labels)
    assert acc == 1.0

