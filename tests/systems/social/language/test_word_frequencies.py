"""Test the `_word_frequencies` module of `skcomplex.systems.social.language`."""

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.social.language import MandelbrotLeastEffortSystem


def _cost_over_entropy(exponent: float, costs: np.ndarray) -> float:
    """Return the mean cost over the entropy of the Boltzmann weights of an exponent."""
    probabilities = np.exp(-exponent * costs)
    probabilities = probabilities / probabilities.sum()
    return float((probabilities * costs).sum() / -(probabilities * np.log(probabilities)).sum())


def test_mandelbrot_least_effort_frequencies() -> None:
    """Test that the frequencies follow a power law of the solved exponent, which minimises cost over entropy."""
    trajectory = simulate(MandelbrotLeastEffortSystem().prepare().system_, until=1.0)
    exponent = float(trajectory.values('language.exponent')[0])
    costs = trajectory.values('words.cost')[0]
    probabilities = trajectory.values('words.probability')[0]
    np.testing.assert_allclose(probabilities, (np.arange(1, costs.size + 1) + 1.0) ** -exponent, atol=1e-9)
    ratio = _cost_over_entropy(exponent, costs)
    assert ratio < _cost_over_entropy(exponent - 0.1, costs)
    assert ratio < _cost_over_entropy(exponent + 0.1, costs)
