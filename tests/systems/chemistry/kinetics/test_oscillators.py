"""Test the `_oscillators` module of `skcomplex.systems.chemistry.kinetics`."""

import numpy as np

from skcomplex.systems.chemistry.kinetics import BrusselatorODESystem, CubicAutocatalysisSystem, OregonatorSystem

from .conftest import _run


def _upward_crossings(times, values, level):
    """Return the interpolated times where a series crosses a level upwards."""
    below, above = values[:-1], values[1:]
    rising = np.flatnonzero((below < level) & (above >= level))
    fractions = (level - below[rising]) / (above[rising] - below[rising])
    return times[rising] + fractions * (times[rising + 1] - times[rising])


def test_oregonator_oscillates():
    """Test that the Oregonator with its standard parameters settles on a relaxation limit cycle of stable period."""
    trajectory = _run(OregonatorSystem(), until=25.0, dt=0.02)
    late = trajectory.times > 5.0
    x = trajectory.get_values('reactor.x')[late]
    periods = np.diff(_upward_crossings(trajectory.times[late], x, 0.1))
    assert len(periods) >= 2
    assert np.ptp(periods) < 0.01 * periods.mean()
    assert x.max() / x.min() > 100


def test_brusselator_ode_has_hopf_bifurcation():
    """Test that the Brusselator rests at `(a, b / a)` below `b = 1 + a**2` and oscillates above it."""
    resting = _run(BrusselatorODESystem(b=1.5), until=60.0, dt=0.5)
    assert np.allclose(
        [resting.get_values('reactor.u')[-1], resting.get_values('reactor.v')[-1]],
        [1.0, 1.5],
        atol=1e-3,
    )
    oscillating = _run(BrusselatorODESystem(b=3.0), until=60.0, dt=0.1)
    assert np.ptp(oscillating.get_values('reactor.u')[oscillating.times > 30.0]) > 1.0


def test_cubic_autocatalysis_is_bistable():
    """Test that cubic autocatalysis settles on the washout or on the reacting state depending on its start."""
    washout = _run(CubicAutocatalysisSystem(a=1.0, b=0.05), until=500.0, dt=5.0)
    reacting = _run(CubicAutocatalysisSystem(a=0.5, b=0.5), until=500.0, dt=5.0)
    assert washout.get_values('reactor.b')[-1] < 1e-3
    assert np.isclose(reacting.get_values('reactor.a')[-1], (1 - np.sqrt(1 - 4 * 0.09**2 / 0.04)) / 2, atol=1e-3)
