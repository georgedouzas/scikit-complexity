"""Test the `_extraction` module of `skcomplex.systems.economics.resources`."""

import numpy as np
import pytest

from skcomplex.systems.economics.resources import FisheryCapitalSystem, OilEconomySystem

from ...conftest import _run_monitored
from ..conftest import _run


def test_oil_economy_capital_rises_and_falls_with_the_resource():
    """Test that the resource only falls, and the capital peaks inside the run and then declines."""
    trajectory = _run(OilEconomySystem(), 100.0, dt=1.0)
    assert np.all(np.diff(trajectory.get_values('oil.resource')) <= 1e-9)
    capital = trajectory.get_values('oil.capital')
    peak = int(np.argmax(capital))
    assert 10 < peak < 90
    assert capital[-1] < capital[peak] / 2


def test_fishery_capital_settles_or_collapses_with_its_technology():
    """Test that the default fleet settles at an equilibrium, and a fleet that keeps high yields depletes the stock."""
    settled = _run(FisheryCapitalSystem(), 200.0, dt=1.0)
    assert np.ptp(settled.get_values('fishery.resource')[-20:]) < 1.0
    assert settled.get_values('fishery.resource')[-1] > 100.0
    efficient = tuple(np.sqrt(np.arange(11) / 10).round(3))
    collapsed = _run(FisheryCapitalSystem(yield_table=efficient), 200.0, dt=1.0)
    assert collapsed.get_values('fishery.resource')[-1] < 1.0
    assert collapsed.get_values('fishery.capital')[-1] < collapsed.get_values('fishery.capital').max() / 10


@pytest.mark.parametrize(('factory', 'label'), [(OilEconomySystem, 'oil'), (FisheryCapitalSystem, 'fishery')])
def test_resource_capital_conserves_its_flows(factory, label):
    """Test that Meadows' resource systems move their resource and capital only through flows."""
    trajectory = _run_monitored(factory().prepare().system_, 1.0, 100)
    assert trajectory.get_values(f'{label}.investment').min() >= 0
    assert trajectory.get_values(f'{label}.depreciation')[0] == pytest.approx(5.0 / 20.0)
