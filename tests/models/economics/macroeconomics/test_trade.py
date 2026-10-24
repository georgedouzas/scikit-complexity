"""Test the `_trade` module of `skcomplex.models.economics.macroeconomics`."""

import pytest

from skcomplex.models import describe
from skcomplex.models.economics.macroeconomics import exchange_rates

from ..conftest import _run


def test_exchange_rates_reproduce_the_floating_and_fixed_stories():
    """Test the stages of a 10 percent rise of home prices under a floating and a fixed exchange rate."""
    floating = _run(built := exchange_rates(), 2, dt=1, observables=describe('exchange_rates').observables_of(built))
    assert floating['home.E'][-1] == pytest.approx(0.805, abs=1e-3)
    assert floating['home.M'][-1] == pytest.approx(30.6)
    assert floating['balance'][-1] == pytest.approx(4.2, abs=0.05)
    assert floating['real_rate'][-1] == pytest.approx(floating['real_rate'][0])
    assert floating['output'][-1] == pytest.approx(103.4)
    fixed = _run(
        built := exchange_rates(floating=False),
        2,
        dt=1,
        observables=describe('exchange_rates').observables_of(built),
    )
    assert set(fixed['home.E'].round(9)) == {round(34 / (32 * 1.2), 9)}
    assert fixed['home.X'][-1] == pytest.approx(0.9 * 32)
    assert fixed['real_rate'][-1] > fixed['real_rate'][0]
