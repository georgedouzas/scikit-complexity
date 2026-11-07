"""Test the `_flows` module of `skcomplex.systems.mathematics.dynamical_systems`."""

import numpy as np
import pytest

from skcomplex.core.analysis import lyapunov
from skcomplex.systems import create_simulation
from skcomplex.systems.mathematics.dynamical_systems import (
    finite_time_blow_up,
    hyperchaotic_rossler,
    ikeda,
    langford,
    nose_hoover,
    sprott_mixed_flow,
    thomas,
)

from ..conftest import _run


def test_jerk_is_bounded():
    """Test that Sprott's jerk equation stays on its bounded attractor."""
    x = create_simulation('sprott_jerk').prepare().run().trajectory_.values('jerk.x')
    assert np.all(np.isfinite(x))
    assert np.abs(x).max() < 10


def test_finite_time_blow_up_follows_the_hyperbola():
    """Test that `x' = x²` follows `x0 / (1 - x0 t)` towards its singularity at `t = 1 / x0`."""
    assert _run(finite_time_blow_up(1.0), 0.9, dt=0.05).values('growth.x')[-1] == pytest.approx(10.0, rel=1e-4)


def test_lorenz_attractor_is_bounded_and_visits_both_lobes():
    """Test that the Lorenz system stays bounded for 50 time units and visits both lobes of its attractor."""
    trajectory = create_simulation('lorenz').prepare().run().trajectory_
    x, z = trajectory.values('lorenz.x'), trajectory.values('lorenz.z')
    assert np.abs(x).max() < 30
    assert z.max() < 60
    late = x[len(x) // 5 :]
    assert (late > 5).any()
    assert (late < -5).any()


def test_nose_hoover_periodic_orbit_closes_and_balances_the_thermostat():
    """Test that the periodic orbit of Datseris and Parlitz returns to its start and every orbit keeps `<y²> = 1`."""
    periodic = _run(nose_hoover(0.0, 1.549934227822898, 0.0), 100, dt=0.05).values('oscillator.y')
    assert np.abs(periodic).max() == pytest.approx(1.549934227822898, rel=1e-4)
    chaotic = _run(nose_hoover(), 1000, dt=0.05).values('oscillator.y')
    assert np.mean(chaotic**2) == pytest.approx(1.0, abs=0.05)


def test_sprott_mixed_flow_conserves_volume_on_the_torus_and_contracts_on_the_attractor():
    """Test that the mean volume rate `2(y + z)` is zero from `(1, 0, 0)` and negative from `(2, 0, 0)`."""
    rates = []
    for x in (1.0, 2.0):
        trajectory = _run(sprott_mixed_flow(x), 1000, dt=0.05)
        rates.append(np.mean(2 * (trajectory.values('flow.y') + trajectory.values('flow.z'))[2000:]))
    assert abs(rates[0]) < 0.01
    assert rates[1] < -0.1


def test_langford_and_thomas_attractors_are_chaotic():
    """Test that the Langford and Thomas flows have a positive and a zero exponent at Dilão's parameters."""
    for system in (langford(), thomas()):
        exponents = lyapunov(system, until=1000).exponents
        assert exponents[0] > 0.01
        assert abs(exponents[1]) < 0.01
        assert exponents[2] < 0


def test_hyperchaotic_rossler_has_two_positive_exponents():
    """Test that the four-dimensional Rössler flow is hyperchaotic, with exponents near 0.11 and 0.02."""
    exponents = lyapunov(hyperchaotic_rossler(), until=2000).exponents
    assert exponents[0] == pytest.approx(0.11, abs=0.02)
    assert exponents[1] == pytest.approx(0.02, abs=0.01)
    assert exponents[3] < 0


def test_ikeda_rest_point_loses_stability_at_a_quarter_turn_of_delay():
    """Test that `x' = a sin x(t - tau)` settles at π for `a tau < π / 2` and oscillates about it beyond."""
    settled = _run(ikeda(2.0, a=0.5, tau=2.0), 200, dt=0.1).values('ikeda.x')
    assert settled[-1] == pytest.approx(np.pi, abs=1e-6)
    oscillating = _run(ikeda(2.0, a=1.0, tau=2.0), 200, dt=0.1).values('ikeda.x')[-300:]
    assert oscillating.min() < np.pi - 1
    assert oscillating.max() > np.pi + 1
