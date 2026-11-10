"""Test the `_kinetic` module of `skcomplex.systems.economics.econophysics`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.economics.econophysics import (
    KineticWealthExchangeSystem,
)


def _final_wealth(until, **options):
    """Return the wealth of every trader at the end of a kinetic exchange run."""
    trajectory = simulate(
        KineticWealthExchangeSystem(500, **options).prepare().system_,
        until=until,
        random_state=0,
        observables=KineticWealthExchangeSystem().get_run_settings()['observables'],
    )
    return trajectory.get_agents('traders', until).values['wealth']


def test_kinetic_exchange_stationary_distributions():
    """Without saving wealth is exponential, with saving it is Gamma of shape `1 + 3 lambda / (1 - lambda)`."""
    exponential = _final_wealth(150)
    assert exponential.mean() == pytest.approx(1.0)
    assert exponential.var() == pytest.approx(1.0, rel=0.3)
    assert np.mean(exponential < np.log(2)) == pytest.approx(0.5, abs=0.06)
    saving = 0.5
    shape = 1 + 3 * saving / (1 - saving)
    gamma = _final_wealth(150, saving=saving)
    assert gamma.var() == pytest.approx(1 / shape, rel=0.2)
    assert np.mean(gamma < 0.1) < 0.01


def test_kinetic_exchange_debt_shifts_exponential():
    """With a debt limit the wealth is exponential shifted to minus the limit, with temperature `<w> + d`."""
    wealth = _final_wealth(300, debt=1.0, amount=0.25)
    assert wealth.min() >= -1.0
    temperature = 2.0
    assert np.mean(wealth < 0) == pytest.approx(1 - np.exp(-1 / temperature), abs=0.07)
    assert wealth.std() == pytest.approx(temperature, rel=0.2)
