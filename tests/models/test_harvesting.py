"""Test the farmer and resource harvesting co-simulation."""

import pytest

from skcomplex.core.runtime import Simulation
from skcomplex.models.economics.resources import harvesting


def test_high_effort_collapses_the_stock():
    """Test that an effort above the growth rate per farmer drives the stock to zero."""
    trajectory = Simulation(harvesting(10, effort=0.15), random_state=0).run(until=100).trajectory_
    assert trajectory['land/stock.n'][-1] < 1e-3


def test_sustainable_effort_reaches_equilibrium():
    """Test that a sustainable effort settles at the equilibrium `K (1 - effort farmers / r)`."""
    trajectory = Simulation(harvesting(10, effort=0.05), random_state=0).run(until=100).trajectory_
    assert trajectory['land/stock.n'][-1] == pytest.approx(500.0, rel=1e-3)
    assert trajectory.agents('village/farmers', 100).values['wealth'].min() > 0
