"""Test the `_oscillators` module of `skcomplex.models.chemistry.kinetics`."""

import numpy as np

from skcomplex.models import describe
from skcomplex.models.chemistry.kinetics import (
    brusselator_ode,
    cubic_autocatalysis,
    oregonator,
)

from .conftest import _run


def _upward_crossings(times, values, level):
    """Return the interpolated times where a series crosses a level upwards."""
    below, above = values[:-1], values[1:]
    rising = np.flatnonzero((below < level) & (above >= level))
    fractions = (level - below[rising]) / (above[rising] - below[rising])
    return times[rising] + fractions * (times[rising + 1] - times[rising])


def test_oregonator_oscillates():
    """Test that the Oregonator with its standard parameters settles on a relaxation limit cycle of stable period."""
    solver = describe('oregonator').defaults['solver']
    trajectory = _run(oregonator(), until=25.0, dt=0.02, solver=solver)
    late = trajectory.times > 5.0
    x = trajectory.values('reactor.x')[late]
    periods = np.diff(_upward_crossings(trajectory.times[late], x, 0.1))
    assert len(periods) >= 2
    assert np.ptp(periods) < 0.01 * periods.mean()
    assert x.max() / x.min() > 100


def test_brusselator_ode_has_hopf_bifurcation():
    """Test that the Brusselator rests at `(a, b / a)` below `b = 1 + a**2` and oscillates above it."""
    resting = _run(brusselator_ode(b=1.5), until=60.0, dt=0.5)
    assert np.allclose([resting.values('reactor.u')[-1], resting.values('reactor.v')[-1]], [1.0, 1.5], atol=1e-3)
    oscillating = _run(brusselator_ode(b=3.0), until=60.0, dt=0.1)
    assert np.ptp(oscillating.values('reactor.u')[oscillating.times > 30.0]) > 1.0


def test_cubic_autocatalysis_is_bistable():
    """Test that cubic autocatalysis settles on the washout or on the reacting state depending on its start."""
    washout = _run(cubic_autocatalysis(a=1.0, b=0.05), until=500.0, dt=5.0)
    reacting = _run(cubic_autocatalysis(a=0.5, b=0.5), until=500.0, dt=5.0)
    assert washout.values('reactor.b')[-1] < 1e-3
    assert np.isclose(reacting.values('reactor.a')[-1], (1 - np.sqrt(1 - 4 * 0.09**2 / 0.04)) / 2, atol=1e-3)
