import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from previs.featuregeneration.advanced import (
    generate_average_word_embeddings,
    generate_tfidf_features,
    horizontal_flip,
)


def test_generate_tfidf_features():
    texts = ['hello world', 'hello']
    features, vocab = generate_tfidf_features(texts)
    assert len(features) == 2
    assert len(vocab) == len(features[0])
    world_index = vocab.index('world')
    # second document does not contain 'world'
    assert features[1][world_index] == 0


def test_generate_average_word_embeddings():
    embeddings = {
        'hello': [1.0, 0.0],
        'world': [0.0, 1.0],
    }
    texts = ['hello world', 'unknown']
    vectors = generate_average_word_embeddings(texts, embeddings)
    assert vectors[0] == [0.5, 0.5]
    # no known words -> zero vector
    assert vectors[1] == [0.0, 0.0]


def test_horizontal_flip():
    img = [[1, 2], [3, 4]]
    flipped = horizontal_flip(img)
    assert flipped == [[2, 1], [4, 3]]
