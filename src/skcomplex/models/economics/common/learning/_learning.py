"""Learning blocks of agent rules: forecasting heuristics and logit choice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from scipy.special import softmax

from .....types import Array

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
    >>> from skcomplex.models.economics.common.learning import heuristics
    >>> heuristics.trend([1.0, 2.0], strength=0.5)
    2.5
"""


def logit_probabilities(utilities: npt.ArrayLike, intensity: float) -> Array:
    """Return the logit choice probabilities `exp(beta U_h) / sum_k exp(beta U_k)`.

    Args:
        utilities:
            The utility of every option, such as past performance of every strategy.

        intensity:
            The intensity of choice `beta`, zero for uniform choice and infinite for the best option.

    Returns:
        probabilities:
            One probability per option.

    Examples:
        >>> from skcomplex.models.economics.common.learning import logit_probabilities
        >>> logit_probabilities([0.0, 0.0], 1.0).tolist()
        [0.5, 0.5]
    """
    values = np.asarray(utilities, dtype=float)
    if np.isinf(intensity):
        best = values == values.max()
        return best / best.sum()
    return np.asarray(softmax(intensity * values), dtype=float)


def logit_choice(utilities: npt.ArrayLike, intensity: float, rng: np.random.Generator) -> int:
    """Return the index of an option drawn with the logit probabilities of its utility.

    Args:
        utilities:
            The utility of every option.

        intensity:
            The intensity of choice.

        rng:
            The random generator, such as `model.rng`.

    Returns:
        index:
            The chosen option.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.models.economics.common.learning import logit_choice
        >>> logit_choice([0.0, 100.0], 10.0, np.random.default_rng(0))
        1
    """
    probabilities = logit_probabilities(utilities, intensity)
    return int(rng.choice(len(probabilities), p=probabilities))
