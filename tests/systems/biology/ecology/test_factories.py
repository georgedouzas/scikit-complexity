"""Test the `_factories` module of `skcomplex.systems.biology.ecology`."""

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.ecology import (
    FibonacciRabbitsSystem,
    LotkaVolterraSystem,
    PenaeidShrimpSystem,
    RickerSystem,
    VaryingCapacityLogisticSystem,
)

from ..conftest import _run_until


def test_lotka_volterra_by_name_takes_predation_rate():
    """Test that the Lotka-Volterra system created with a predation rate oscillates around its center."""
    trajectory = _run_until(LotkaVolterraSystem(a=0.2), 50, dt=0.1)
    assert trajectory.get_values('prey.x').min() < 0.4 / 0.1 < trajectory.get_values('prey.x').max()


def test_varying_capacity_settles_on_cycle_of_the_season():
    """Test that logistic growth under a periodic capacity settles on a cycle of the same period."""
    trajectory = _run_until(VaryingCapacityLogisticSystem(), 20, dt=0.05)
    population = trajectory.get_values('population.x')
    assert population[-1] == pytest.approx(population[-21], rel=1e-3)
    assert population[-20:].max() - population[-20:].min() > 10


def test_fibonacci_rabbits():
    """Test that Fibonacci's rabbits double their growth like the Fibonacci numbers."""
    assert simulate(FibonacciRabbitsSystem(), until=9).get_values(
        'rabbits.pairs',
    ).tolist() == [
        1,
        1,
        2,
        3,
        5,
        8,
        13,
        21,
        34,
        55,
    ]


def test_ricker_settles_on_its_fixed_point_and_doubles_its_period():
    """Test that Ricker's map settles on `ln r` below `r = e^2` and on a two-cycle just above it."""
    stable = _run_until(RickerSystem(r=5.0), 200, dt=1).get_values('population.x')
    assert stable[-1] == pytest.approx(np.log(5.0))
    cycle = _run_until(RickerSystem(r=9.0), 200, dt=1).get_values('population.x')
    assert cycle[-1] == pytest.approx(cycle[-3])
    assert abs(cycle[-1] - cycle[-2]) > 1.0


def test_penaeid_shrimp_length_follows_von_bertalanffy():
    """Test that the carapace length follows `L_inf (1 - exp(-K t))` and the gut settles near `I / A_e`."""
    trajectory = _run_until(PenaeidShrimpSystem(), 60, dt=1.0)
    times = trajectory.times
    np.testing.assert_allclose(trajectory.get_values('shrimp.length'), 29.9 * (1 - np.exp(-0.013 * times)), rtol=1e-4)
    assert 0.3 < trajectory.get_values('shrimp.gut')[-1] < 0.6
    assert np.all(np.diff(trajectory.get_values('shrimp.gonads')) > 0)
