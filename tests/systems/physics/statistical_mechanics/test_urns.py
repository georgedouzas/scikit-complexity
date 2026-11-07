"""Test the `_urns` module of `skcomplex.systems.physics.statistical_mechanics`."""

import math

import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.statistical_mechanics import EhrenfestUrnSystem


def test_ehrenfest_urn_relaxes_to_the_binomial():
    """Test that the urn started full relaxes to the binomial distribution of mean N / 2 and variance N / 4."""
    counts = np.asarray(
        simulate(EhrenfestUrnSystem(40), until=20000, random_state=0).values('urn.n'),
    )
    assert counts[0] == 40
    np.testing.assert_allclose(counts[:30], 20 + 20 * (1 - 2 / 40) ** np.arange(30), atol=8)
    frequencies = np.bincount(counts[400:].astype(int), minlength=41) / counts[400:].size
    binomial = np.array([math.comb(40, k) for k in range(41)]) / 2**40
    assert np.abs(frequencies - binomial).max() < 0.02
    np.testing.assert_allclose([counts[400:].mean(), counts[400:].var()], [20, 10], rtol=0.1)
