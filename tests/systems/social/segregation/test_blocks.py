"""Test the `_blocks` module of `skcomplex.systems.social.segregation`."""

import numpy as np

from skcomplex.systems.social.segregation import StochasticSchellingSystem

from ..conftest import _run, _values


def _block_densities(trajectory):
    """Return the sorted densities of the 16 blocks of 25 cells of the stochastic Schelling system."""
    blocks = _values(trajectory, 'dwellers', 'block').astype(int)
    return np.bincount(blocks, minlength=16) / 25


def test_stochastic_schelling_segregates_above_critical_beta():
    """Test that block densities split towards 0.07 and 0.93 at `beta = 3`, and stay near 0.5 at `beta = 1`."""
    hot = _block_densities(_run(StochasticSchellingSystem(beta=1.0, random_state=0), 100))
    cold = _block_densities(_run(StochasticSchellingSystem(beta=3.0, random_state=0), 100))
    assert cold.std() > 2 * hot.std()
    assert np.abs(hot - 0.5).max() < 0.4
    assert cold.min() < 0.2
    assert cold.max() > 0.8
