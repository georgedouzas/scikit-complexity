"""Test the `_factories` module of `skcomplex.systems.biology.ecology`."""

import pytest

from skcomplex.systems import create_simulation
from skcomplex.systems.biology.ecology import (
    lotka_volterra,
    varying_capacity_logistic,
)

from ..conftest import _run_until


def test_lotka_volterra_by_name_takes_predation_rate():
    """Test that the Lotka-Volterra system created with a predation rate oscillates around its center."""
    trajectory = _run_until(lotka_volterra(a=0.2), 50, dt=0.1)
    assert trajectory.values('prey.n').min() < 0.4 / 0.1 < trajectory.values('prey.n').max()


def test_varying_capacity_settles_on_cycle_of_the_season():
    """Test that logistic growth under a periodic capacity settles on a cycle of the same period."""
    trajectory = _run_until(varying_capacity_logistic(), 20, dt=0.05)
    population = trajectory.values('population.n')
    assert population[-1] == pytest.approx(population[-21], rel=1e-3)
    assert population[-20:].max() - population[-20:].min() > 10


def test_fibonacci_rabbits():
    """Test that Fibonacci's rabbits double their growth like the Fibonacci numbers."""
    assert create_simulation('fibonacci_rabbits').prepare().run(until=9).trajectory_.values(
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
