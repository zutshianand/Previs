import os
import sys
import types

# Ensure the repository root is on the path so ``previs`` can be imported.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub external dependencies that are heavy or unavailable in the execution
# environment.  ``TextProcessingUtils`` imports ``gensim`` and ``wordcloud`` on
# module import.  Provide lightweight placeholders so the module can be
# imported without the real packages.
gensim_stub = types.ModuleType("gensim")
gensim_stub.corpora = types.ModuleType("corpora")
gensim_stub.models = types.ModuleType("models")
gensim_stub.similarities = types.ModuleType("similarities")
fasttext_stub = types.ModuleType("fasttext")
class _FT:  # pragma: no cover - simple placeholder
    pass
fasttext_stub.FastText = _FT
gensim_stub.models.fasttext = fasttext_stub
sys.modules.setdefault("gensim", gensim_stub)
sys.modules.setdefault("gensim.corpora", gensim_stub.corpora)
sys.modules.setdefault("gensim.models", gensim_stub.models)
sys.modules.setdefault("gensim.models.fasttext", fasttext_stub)
sys.modules.setdefault("gensim.similarities", gensim_stub.similarities)

wordcloud_stub = types.ModuleType("wordcloud")
wordcloud_stub.STOPWORDS = set()
sys.modules.setdefault("wordcloud", wordcloud_stub)

# Stub out the heavy ``TextProcessor`` dependency as well
processors_mod = types.ModuleType("previs.processors")
tp_mod = types.ModuleType("previs.processors.TextProcessor")
class _TP:  # pragma: no cover - simple placeholder
    def process(self, text):
        return text
tp_mod.TextProcessor = _TP
sys.modules.setdefault("previs.processors", processors_mod)
sys.modules.setdefault("previs.processors.TextProcessor", tp_mod)

from previs.util import TextProcessingUtils as tpu


class DummyTextProcessor:
    """Minimal stand-in for :class:`TextProcessor`.

    The real implementation performs heavy NLP steps which are unnecessary for
    this unit test.  This dummy simply returns the lower-cased text and allows
    the test to focus on verifying that the FastText model is instantiated
    correctly.
    """

    def process(self, text: str) -> str:  # pragma: no cover - trivial
        return text.lower()


class DummyFastText:
    """Simplified replacement for :class:`gensim.models.FastText`.

    It only captures the ``vector_size`` argument and provides the minimal API
    used by ``get_distance_between_sentences``.
    """

    def __init__(self, *, vector_size):
        # Fail the test if the function still uses the deprecated ``size``
        # keyword by not accepting it here.
        assert vector_size == 100
        # Minimal attributes referenced during training
        self.epochs = 1
        self.corpus_count = 0
        self.corpus_total_words = 0

    def build_vocab(self, sentences):  # pragma: no cover - simple stub
        self.sentences = sentences

    def train(self, sentences, epochs, total_examples, total_words):  # pragma: no cover - simple stub
        pass

    def wmdistance(self, s1, s2):  # pragma: no cover - simple stub
        return 0.123


def dummy_build_clean_corpus(df, col_name):  # pragma: no cover - simple stub
    return ["foo bar"], col_name


def test_get_distance_between_sentences_uses_vector_size(monkeypatch):
    """Ensure ``get_distance_between_sentences`` instantiates FastText using
    the modern ``vector_size`` argument and returns the model's distance."""

    # Replace heavy dependencies with lightweight stand-ins
    monkeypatch.setattr(tpu, "TextProcessor", DummyTextProcessor)
    monkeypatch.setattr(tpu, "FT_gensim", DummyFastText)
    monkeypatch.setattr(tpu, "build_clean_corpus", dummy_build_clean_corpus)

    dummy_df = object()
    result = tpu.get_distance_between_sentences(dummy_df, "text", "A", "B")

    assert result == 0.123
