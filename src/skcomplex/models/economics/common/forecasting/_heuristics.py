"""Define the forecasting heuristics of agent rules, functions of the past values of a series."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np

# The values a trend or an anchor reads, the last two.
TREND_LENGTH = 2


def _adaptive(history: Sequence[float], previous: float, weight: float = 0.65) -> float:
    """Return the adaptive forecast `w x(t) + (1 - w) f(t)` of the last value and the previous forecast."""
    return float(weight * history[-1] + (1 - weight) * previous)


def _trend(history: Sequence[float], strength: float = 1.3) -> float:
    """Return the trend-following forecast `x(t) + g (x(t) - x(t - 1))`."""
    if len(history) < TREND_LENGTH:
        return float(history[-1])
    return float(history[-1] + strength * (history[-1] - history[-2]))


def _anchoring(history: Sequence[float]) -> float:
    """Return the anchoring-and-adjustment forecast `(mean + x(t)) / 2 + (x(t) - x(t - 1))`."""
    if len(history) < TREND_LENGTH:
        return float(history[-1])
    return float((np.mean(history) + history[-1]) / 2 + history[-1] - history[-2])


def _fundamental(history: Sequence[float], fundamental: float = 0.0, reversion: float = 0.0) -> float:
    """Return the fundamentalist forecast `F + v (x(t) - F)`."""
    return float(fundamental + reversion * (history[-1] - fundamental))


heuristics = SimpleNamespace(adaptive=_adaptive, trend=_trend, anchoring=_anchoring, fundamental=_fundamental)
heuristics.__doc__ = """Forecasting heuristics of Anufriev and Hommes, functions of the past values, the last one last.

- `adaptive(history, previous, weight)` gives `w x(t) + (1 - w) f(t)`, with `f(t)` the previous forecast.
- `trend(history, strength)` gives `x(t) + g (x(t) - x(t - 1))`.
- `anchoring(history)` gives `(mean + x(t)) / 2 + (x(t) - x(t - 1))`.
- `fundamental(history, fundamental, reversion)` gives `F + v (x(t) - F)`.

Examples:
    >>> from skcomplex.models.economics.common.forecasting import heuristics
    >>> heuristics.trend([1.0, 2.0], strength=0.5)
    2.5
"""
