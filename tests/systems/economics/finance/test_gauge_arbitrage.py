"""Test the `_gauge_arbitrage` module of `skcomplex.systems.economics.finance`."""

import numpy as np

from skcomplex.core.engine.compile import compile_system
from skcomplex.systems import create_simulation
from skcomplex.systems.economics.finance import gauge_arbitrage_market


def test_market_conserves_the_good_and_the_gauss_law():
    """Test that the total good stays within 1e-10 relative error, and the Gauss residual below 1e-8."""
    trajectory = create_simulation('gauge_arbitrage_market').prepare().run().trajectory_
    total = trajectory.values('total_good')
    assert np.abs(total - total[0]).max() / total[0] < 1e-10
    assert trajectory.values('gauss_residual').max() < 1e-8


def test_market_moves_the_good_and_the_exchange_rates():
    """Test that the good spreads out and the potentials leave their initial value."""
    trajectory = create_simulation('gauge_arbitrage_market').prepare().run().trajectory_
    density = trajectory.values('sites.rho')
    assert density[-1].std() < density[0].std()
    assert np.abs(trajectory.values('links.A')[-1]).max() > 0.01


def test_current_and_field_follow_the_signs_of_the_book():
    """Test the rates against `J = -D (d rho - q A rho)`, `rho_t = -div J`, `A_t = E`, and `E_t = -curl curl A - J`."""
    D, q = 0.5, 0.2
    system = gauge_arbitrage_market(4, D=D, q=q)
    problem = compile_system(system)
    space = system.elements[0].space
    edges, faces = space.incidences_[:2]
    state = problem.initial_state.copy()
    rng = np.random.default_rng(1)
    n_sites, n_links = edges.shape[1], edges.shape[0]
    state[n_sites : n_sites + n_links] = rng.normal(size=n_links)
    density = state[:n_sites]
    potential, field = state[n_sites : n_sites + n_links], state[n_sites + n_links :]
    rates = problem.drift(0.0, state, problem.initial_parameters)
    middle = np.abs(edges) @ density / 2
    current = -D * (edges @ density - q * potential * middle)
    assert np.allclose(rates[:n_sites], edges.T @ current)
    assert np.allclose(rates[n_sites : n_sites + n_links], field)
    assert np.allclose(rates[n_sites + n_links :], -(faces.T @ (faces @ potential)) - current)
