"""Test the `_metaphoria` module of `skcomplex.systems.economics.macroeconomics`."""

import numpy as np
import pytest

from skcomplex.systems.economics.macroeconomics import MetaphoriaSystem

from ..conftest import _run_observed


def _metaphoria(until, invariants=None, **arguments):
    """Return the trajectory of Metaphoria with its standard observables."""
    system = MetaphoriaSystem(100, random_state=0, **arguments)
    return _run_observed(system.prepare().system_, until, system.get_run_settings()['observables'], invariants)


def test_metaphoria_hunter_gatherers_thrive_where_money_dies_out():
    """Hunter-gatherers keep a stable population on land where the monetary population dies out, as in Gooding."""
    gatherers = _metaphoria(150).values('population')
    monetary = _metaphoria(150, mode='monetary').values('population')
    assert gatherers[100:].min() > 50
    assert monetary[-1] < gatherers[-1] / 4


def test_metaphoria_monetary_money_is_stock_flow_consistent():
    """With abundant food the monetary economy lives with starvation, and all money held is matched by loans."""
    trajectory = _metaphoria(100, {'bank_balance': 1e-6}, mode='monetary', max_food=50.0, max_resistance=0.0)
    assert trajectory.values('population')[-1] > 100
    assert np.nansum(trajectory.values('census.starvations')) > 0
    assert trajectory.values('money_gini')[-1] > 0.3
    holders = [trajectory.get_agents(group, 100).values for group in ('people', 'businesses', 'bank')]
    money = sum(np.sum(values['money']) for values in holders)
    loans = sum(np.sum(values['loans']) for values in holders)
    assert abs(money) < 1e-6 * np.sum(holders[2]['loans'])
    assert abs(loans) < 1e-6 * np.sum(holders[2]['loans'])
    assert np.sum(holders[2]['loans']) > 0


def test_metaphoria_rejects_unknown_mode():
    """An unknown mode raises an error."""
    with pytest.raises(ValueError, match='mode'):
        MetaphoriaSystem(mode='barter').prepare()
