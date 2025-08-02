"""Interactive visualisation helpers.

This module exposes minimal wrappers around popular interactive plotting
libraries.  The functions import the required backends lazily so that the
core :mod:`previs` package does not hard depend on these optional
dependencies.  Each helper will raise an informative :class:`ImportError`
if the library is not available.

The helpers return the figure object created by the underlying library so
that callers can further tweak or display them as needed.
"""

from __future__ import annotations

from typing import Any, Iterable


def plotly_scatter(x: Iterable[float], y: Iterable[float]) -> Any:
    """Create an interactive scatter plot using :mod:`plotly`.

    Parameters
    ----------
    x, y:
        Numeric iterables describing the coordinates of the points.

    Returns
    -------
    Any
        The resulting Plotly figure.

    Raises
    ------
    ImportError
        If Plotly is not installed.
    """

    try:
        import plotly.express as px  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Plotly is required for interactive plotting. Install it via 'pip "
            "install plotly'."
        ) from exc

    fig = px.scatter(x=x, y=y)
    return fig


def bokeh_scatter(x: Iterable[float], y: Iterable[float]) -> Any:
    """Create an interactive scatter plot using :mod:`bokeh`.

    Parameters are identical to :func:`plotly_scatter`.
    """

    try:
        from bokeh.plotting import figure
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Bokeh is required for interactive plotting. Install it via 'pip "
            "install bokeh'."
        ) from exc

    p = figure()
    p.scatter(x=list(x), y=list(y))
    return p


__all__ = ["plotly_scatter", "bokeh_scatter"]

