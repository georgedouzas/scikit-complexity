"""Test the `_exchange` module of `skcomplex.models.economics.econophysics`."""

import numpy as np
import pytest

from skcomplex.core.runtime import Simulation
from skcomplex.models.economics.econophysics import boltzmann_wealth

from ..conftest import _agents, _run


def test_boltzmann_wealth_stays_constant():
    """Test that the total wealth of all traders equals its initial value at every frame."""
    trajectory = Simulation(boltzmann_wealth(100), random_state=0).run(until=50).trajectory_
    totals = {float(trajectory.agents('traders', frame).values['wealth'].sum()) for frame in range(51)}
    assert totals == {100.0}


def test_kinetic_wealth_exchange_conserves_money_and_approaches_an_exponential():
    """Test that random money exchange keeps the total constant and approaches the exponential distribution."""
    trajectory = _run(boltzmann_wealth(200), 200)
    totals = {float(_agents(trajectory, 'traders', frame)['wealth'].sum()) for frame in range(201)}
    assert totals == {200.0}
    wealth = np.concatenate([_agents(trajectory, 'traders', frame)['wealth'] for frame in range(100, 201, 10)])
    for k in (1, 2, 3):
        assert np.mean(wealth >= k) == pytest.approx(0.5**k, abs=0.06)
    assert np.var(wealth) == pytest.approx(2.0, rel=0.25)
