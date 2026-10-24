"""Test the `_barter_networks` module of `skcomplex.models.economics.markets`."""

import numpy as np
import pytest

from skcomplex.models import describe
from skcomplex.models.economics.markets import wilhite_trade_networks

from ..conftest import _run_observed


def _barter(topology):
    """Return the trajectory of Wilhite's barter of 100 traders in 5 groups on one topology."""
    system = wilhite_trade_networks(100, topology=topology, n_groups=5, shortcuts=3, random_state=1)
    return _run_observed(system, 30, describe('wilhite_trade_networks').observables_of(system))


def test_wilhite_trade_networks_small_world_converges_with_few_searches():
    """Isolated groups keep different prices, a small world unifies them with a fraction of the complete searches."""
    isolated, small_world, complete = (_barter(topology) for topology in ('isolated', 'small_world', 'complete'))
    assert small_world['dispersion'][-1] < 0.1
    assert isolated['dispersion'][-1] > 2 * small_world['dispersion'][-1]
    assert complete['barter.searches'][1:].sum() > 4 * small_world['barter.searches'][1:].sum()
    goods = [small_world.agents('traders', frame).values for frame in (0, 30)]
    for good in ('good1', 'good2'):
        np.testing.assert_allclose(np.sum(goods[1][good]), np.sum(goods[0][good]))
    assert small_world['utility'][-1] > small_world['utility'][0]


def test_wilhite_trade_networks_rejects_unknown_topology():
    """An unknown topology raises an error."""
    with pytest.raises(ValueError, match='topology'):
        wilhite_trade_networks(10, topology='ring')
