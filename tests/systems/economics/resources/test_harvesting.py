"""Test the `_harvesting` module of `skcomplex.systems.economics.resources`."""

import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.resources import HarvestingSystem


def test_high_effort_collapses_the_stock():
    """Test that an effort above the growth rate per farmer drives the stock to zero."""
    trajectory = simulate(HarvestingSystem(10, effort=0.15).prepare().system_, until=100, random_state=0)
    assert trajectory.values('land/stock.n')[-1] < 1e-3


def test_sustainable_effort_reaches_equilibrium():
    """Test that a sustainable effort settles at the equilibrium `K (1 - effort farmers / r)`."""
    trajectory = simulate(HarvestingSystem(10, effort=0.05).prepare().system_, until=100, random_state=0)
    assert trajectory.values('land/stock.n')[-1] == pytest.approx(500.0, rel=1e-3)
    assert trajectory.get_agents('village/farmers', 100).values['wealth'].min() > 0
