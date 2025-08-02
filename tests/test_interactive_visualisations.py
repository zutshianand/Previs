import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from previs.visualisations import interactive


def test_plotly_scatter_requires_dependency():
    with pytest.raises(ImportError):
        interactive.plotly_scatter([1], [1])


def test_bokeh_scatter_requires_dependency():
    with pytest.raises(ImportError):
        interactive.bokeh_scatter([1], [1])

