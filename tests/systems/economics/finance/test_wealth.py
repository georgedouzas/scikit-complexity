"""Test the `_wealth` module of `skcomplex.systems.economics.finance`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.systems.economics.finance import (
    BouchaudMezardWealthSystem,
)

from ..conftest import _run


def test_bouchaud_mezard_exchange_conserves_and_equalizes_wealth():
    """Test that exchange without returns conserves the total and relaxes wealth to the mean at rate `J N / (N - 1)`."""
    wealth = np.linspace(0.5, 1.5, 20)
    trajectory = _run(
        BouchaudMezardWealthSystem(20, sigma=0.0, wealth=wealth).prepare().system_,
        1.0,
        dt=0.1,
        observables=BouchaudMezardWealthSystem().get_run_settings()['observables'],
    )
    final = trajectory.values('agents.W')[-1]
    assert final.sum() == pytest.approx(20.0)
    assert np.allclose(final - 1.0, (wealth - 1.0) * np.exp(-20 / 19), atol=2e-3)
    noisy = _run(
        BouchaudMezardWealthSystem(20).prepare().system_,
        1.0,
        dt=0.1,
        observables=BouchaudMezardWealthSystem().get_run_settings()['observables'],
    ).values('agents.W')
    assert np.all(noisy > 0)


def test_bouchaud_mezard_exchanges_with_the_mean():
    """Test that the agents exchange with the group mean as with the complete-graph Laplacian, conserving wealth."""
    wealth = np.random.default_rng(0).uniform(0.5, 1.5, 50)
    built = BouchaudMezardWealthSystem(50, sigma=0.0, wealth=wealth).prepare().system_
    observables = BouchaudMezardWealthSystem().get_run_settings()['observables']
    mean_field = PassCompiler(observables=observables).prepare(built).run().problem_
    built = BouchaudMezardWealthSystem(50, graph=nx.complete_graph(50), sigma=0.0, wealth=wealth).prepare().system_
    observables = BouchaudMezardWealthSystem().get_run_settings()['observables']
    complete = PassCompiler(observables=observables).prepare(built).run().problem_
    exchange = mean_field.drift(0.0, mean_field.initial_state, mean_field.initial_parameters)
    np.testing.assert_allclose(
        exchange,
        complete.drift(0.0, complete.initial_state, complete.initial_parameters),
        atol=1e-12,
    )
    assert abs(exchange.sum()) < 1e-12
