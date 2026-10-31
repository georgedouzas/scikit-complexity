"""Test the `_learning` module of `skcomplex.models.economics.common.learning`."""

import math

import numpy as np

from skcomplex.models.economics.common.learning import heuristics, logit_choice, logit_probabilities


def test_heuristics_and_logit():
    """Forecasting heuristics and logit choice follow their formulas."""
    assert heuristics.adaptive([2.0], 1.0, weight=0.5) == 1.5
    assert heuristics.trend([1.0, 2.0], strength=1.0) == 3.0
    assert heuristics.anchoring([1.0, 3.0]) == 4.5
    assert heuristics.fundamental([3.0], fundamental=1.0, reversion=0.5) == 2.0
    probabilities = logit_probabilities([1.0, 0.0], np.log(3.0))
    np.testing.assert_allclose(probabilities, [0.75, 0.25])
    assert logit_probabilities([1.0, 2.0], math.inf).tolist() == [0.0, 1.0]
    assert logit_choice([0.0, 50.0], 1.0, np.random.default_rng(0)) == 1
