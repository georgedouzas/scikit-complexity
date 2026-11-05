"""Test the `_energy_balance` module of `skcomplex.systems.physics.climate`."""

import numpy as np

from skcomplex.systems import describe
from skcomplex.systems.physics.climate import (
    budyko_sellers,
    daisyworld,
    energy_balance,
)

from ..conftest import _run_default


def test_ice_albedo_bistability():
    """Test that the energy balance settles in a warm and in a frozen climate at the same solar constant."""
    warm = _run_default(energy_balance(T=300.0), until=1000.0, dt=10.0).values('earth.T')[-1]
    cold = _run_default(energy_balance(T=220.0), until=1000.0, dt=10.0).values('earth.T')[-1]
    assert abs(warm - 286.4) < 0.5
    assert abs(cold - 231.7) < 0.5


def test_budyko_sellers_has_ice_caps_and_a_snowball():
    """Test that present insolation keeps both a climate with polar ice caps and a snowball Earth."""
    capped = _run_default(budyko_sellers(18), until=40.0, dt=4.0).values('bands.T')[-1]
    snowball = _run_default(budyko_sellers(18, T=-50.0), until=40.0, dt=4.0).values('bands.T')[-1]
    assert capped[0] > 20.0 > -10.0 > capped[-1]
    assert np.all(np.diff(capped) < 0)
    assert np.all(snowball < -10.0)


def test_daisyworld_regulates_temperature():
    """Test that white and black daisies coexist and bring the temperature closer to their optimum than bare ground."""
    planet = _run_default(
        built := daisyworld(),
        until=100.0,
        dt=1.0,
        observables=describe('daisyworld').observables_of(built),
    )
    bare = _run_default(
        built := daisyworld(white=0.0, black=0.0),
        until=1.0,
        dt=1.0,
        observables=describe('daisyworld').observables_of(built),
    )
    assert planet.values('planet.white')[-1] > 0.1
    assert planet.values('planet.black')[-1] > 0.1
    assert abs(planet.values('temperature')[-1] - 295.5) < abs(bare.values('temperature')[-1] - 295.5) - 3.0
