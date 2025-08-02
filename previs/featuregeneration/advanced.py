"""Advanced feature generation utilities.

This module provides lightweight helpers for generating features from text and
images without relying on heavy third‑party dependencies.  The functions are
minimal yet demonstrate techniques such as TF–IDF, word embeddings and image
augmentation.
"""

from collections import Counter
import math
from typing import Dict, Iterable, List, Sequence


def generate_tfidf_features(
    texts: Sequence[str],
    max_features: int | None = None,
):
    """Compute TF–IDF features for ``texts``.

    Parameters
    ----------
    texts:
        Iterable of text documents.
    max_features:
        Optional limit on the number of vocabulary terms to keep.  The most
        frequent terms across the corpus are retained when a limit is set.

    Returns
    -------
    features:
        List of lists where each inner list contains the TF–IDF values for a
        document.  The values correspond to the order of ``vocabulary``.
    vocabulary:
        List of vocabulary terms.
    """

    tokenised = [str(t or "").split() for t in texts]

    # Build vocabulary based on frequency
    vocab_counter = Counter()
    for doc in tokenised:
        vocab_counter.update(doc)
    vocabulary = [w for w, _ in vocab_counter.most_common(max_features)]

    # Document frequency for IDF
    doc_freq = {term: 0 for term in vocabulary}
    for doc in tokenised:
        seen = set(doc)
        for term in seen:
            if term in doc_freq:
                doc_freq[term] += 1

    n_docs = len(tokenised)
    idf = {term: math.log((1 + n_docs) / (1 + df)) + 1 for term, df in doc_freq.items()}

    # Compute tfidf rows
    features: List[List[float]] = []
    for doc in tokenised:
        counts = Counter(doc)
        max_count = max(counts.values()) if counts else 1
        row = []
        for term in vocabulary:
            tf = counts.get(term, 0) / max_count
            row.append(tf * idf[term])
        features.append(row)

    return features, vocabulary


def generate_average_word_embeddings(
    texts: Sequence[str],
    embeddings: Dict[str, Iterable[float]],
) -> List[List[float]]:
    """Return the mean embedding vector for each text.

    Parameters
    ----------
    texts:
        Iterable of text documents.
    embeddings:
        Mapping from token to its embedding vector (all of equal length).

    Returns
    -------
    vectors:
        List of averaged embedding vectors.  Missing words are ignored and a
        zero vector is used when none of the tokens are found in ``embeddings``.
    """

    if not embeddings:
        return [[] for _ in texts]

    vector_size = len(next(iter(embeddings.values())))
    result: List[List[float]] = []
    for text in texts:
        tokens = str(text or "").split()
        vecs = [list(map(float, embeddings[t])) for t in tokens if t in embeddings]
        if vecs:
            avg = [sum(vals) / len(vecs) for vals in zip(*vecs)]
        else:
            avg = [0.0] * vector_size
        result.append(avg)
    return result


def horizontal_flip(image: List[List[float]]) -> List[List[float]]:
    """Return a horizontally flipped copy of ``image`` represented as lists.

    ``image`` is assumed to be a 2D array-like structure implemented as a list
    of lists.  Each inner list represents a row of pixels.
    """

    return [row[::-1] for row in image]

