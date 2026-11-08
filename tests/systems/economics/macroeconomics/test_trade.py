"""Test the `_trade` module of `skcomplex.systems.economics.macroeconomics`."""

import pytest

from skcomplex.systems.economics.macroeconomics import ExchangeRatesSystem

from ..conftest import _run


def test_exchange_rates_reproduce_the_floating_and_fixed_stories():
    """Test the stages of a 10 percent rise of home prices under a floating and a fixed exchange rate."""
    floating = _run(
        ExchangeRatesSystem().prepare().system_,
        2,
        dt=1,
        observables=ExchangeRatesSystem().get_run_settings()['observables'],
    )
    assert floating.values('home.E')[-1] == pytest.approx(0.805, abs=1e-3)
    assert floating.values('home.M')[-1] == pytest.approx(30.6)
    assert floating.values('balance')[-1] == pytest.approx(4.2, abs=0.05)
    assert floating.values('real_rate')[-1] == pytest.approx(floating.values('real_rate')[0])
    assert floating.values('output')[-1] == pytest.approx(103.4)
    fixed = _run(
        ExchangeRatesSystem(floating=False).prepare().system_,
        2,
        dt=1,
        observables=ExchangeRatesSystem().get_run_settings()['observables'],
    )
    assert set(fixed.values('home.E').round(9)) == {round(34 / (32 * 1.2), 9)}
    assert fixed.values('home.X')[-1] == pytest.approx(0.9 * 32)
    assert fixed.values('real_rate')[-1] > fixed.values('real_rate')[0]
