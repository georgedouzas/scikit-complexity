"""Test the `_learning` module of `skcomplex.models.economics.common.learning`."""

import math

import numpy as np
import pytest

from skcomplex.models.economics.common.learning import (
    evolve,
    heuristics,
    logit_choice,
    logit_probabilities,
)


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


def test_evolve_operators():
    """Evolution keeps elites, mutates bits, and rejects unknown selections."""
    rng = np.random.default_rng(0)
    genomes = rng.integers(0, 2, (20, 8))
    fitness = genomes.sum(axis=1).astype(float)
    child = evolve(genomes, fitness, rng, elite=2)
    assert child.shape == genomes.shape
    assert set(np.unique(child)) <= {0.0, 1.0}
    np.testing.assert_array_equal(child[0], genomes[np.argmax(fitness)])
    real = evolve(rng.normal(size=(6, 3)), np.arange(6.0), rng, select='roulette', alphabet=None, mutation=1.0)
    assert real.shape == (6, 3)
    with pytest.raises(ValueError, match='selection'):
        evolve(genomes, fitness, rng, select='lottery')
