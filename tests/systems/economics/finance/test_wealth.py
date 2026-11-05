"""Test the `_wealth` module of `skcomplex.systems.economics.finance`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.engine.compile import compile_system
from skcomplex.systems import describe
from skcomplex.systems.economics.finance import (
    bouchaud_mezard_wealth,
)

from ..conftest import _run


def test_bouchaud_mezard_exchange_conserves_and_equalizes_wealth():
    """Test that exchange without returns conserves the total and relaxes wealth to the mean at rate `J N / (N - 1)`."""
    wealth = np.linspace(0.5, 1.5, 20)
    trajectory = _run(
        built := bouchaud_mezard_wealth(20, sigma=0.0, wealth=wealth),
        1.0,
        dt=0.1,
        observables=describe('bouchaud_mezard_wealth').observables_of(built),
    )
    final = trajectory.values('agents.W')[-1]
    assert final.sum() == pytest.approx(20.0)
    assert np.allclose(final - 1.0, (wealth - 1.0) * np.exp(-20 / 19), atol=2e-3)
    noisy = _run(
        built := bouchaud_mezard_wealth(20),
        1.0,
        dt=0.1,
        observables=describe('bouchaud_mezard_wealth').observables_of(built),
    ).values('agents.W')
    assert np.all(noisy > 0)


def test_bouchaud_mezard_exchanges_with_the_mean():
    """Test that the agents exchange with the group mean as with the complete-graph Laplacian, conserving wealth."""
    wealth = np.random.default_rng(0).uniform(0.5, 1.5, 50)
    mean_field = compile_system(
        built := bouchaud_mezard_wealth(50, sigma=0.0, wealth=wealth),
        observables=describe('bouchaud_mezard_wealth').observables_of(built),
    )
    complete = compile_system(
        built := bouchaud_mezard_wealth(50, graph=nx.complete_graph(50), sigma=0.0, wealth=wealth),
        observables=describe('bouchaud_mezard_wealth').observables_of(built),
    )
    exchange = mean_field.drift(0.0, mean_field.initial_state, mean_field.initial_parameters)
    np.testing.assert_allclose(
        exchange,
        complete.drift(0.0, complete.initial_state, complete.initial_parameters),
        atol=1e-12,
    )
    assert abs(exchange.sum()) < 1e-12
