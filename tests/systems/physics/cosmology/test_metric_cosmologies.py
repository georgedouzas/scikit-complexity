"""Test the `_metric_cosmologies` module of `skcomplex.systems.physics.cosmology`."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.systems.physics.cosmology import flrw_cosmology, oppenheimer_snyder_collapse

from ..conftest import _run, _series

ACCURATE = ScipyODESolver(method='DOP853', rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    ('kwargs', 'exact'),
    [
        ({'omega_matter': 1.0, 'omega_lambda': 0.0}, lambda t: (1.5 * t + 0.5**1.5) ** (2 / 3)),
        ({'omega_matter': 0.0, 'omega_radiation': 1.0, 'omega_lambda': 0.0}, lambda t: np.sqrt(2 * t + 0.25)),
        ({'omega_matter': 0.0, 'omega_lambda': 1.0}, lambda t: 0.5 * np.exp(t)),
        ({'omega_matter': 0.0, 'omega_lambda': 0.0}, lambda t: 0.5 + t),
    ],
)
def test_scale_factor_follows_the_analytic_friedmann_solutions(kwargs, exact):
    """Test the scale factor of dust `t^(2/3)`, radiation `t^(1/2)`, vacuum `e^(H t)`, and the empty Milne universe."""
    trajectory = _run(flrw_cosmology(0.5, **kwargs), 2.0, 0.1, ACCURATE)
    (a,) = _series(trajectory, 'universe.a')
    assert np.ravel(a) == pytest.approx(exact(np.asarray(trajectory.times)), rel=1e-8)


def test_mixed_universe_keeps_the_friedmann_constraint_and_accelerates_late():
    """Test that matter, radiation, vacuum, and curvature keep `(a'/a)² = H0² Σ Ω a^-n` and accelerate at late times."""
    trajectory = _run(
        flrw_cosmology(0.3, omega_matter=0.3, omega_radiation=0.01, omega_lambda=0.8),
        3.0,
        0.05,
        ACCURATE,
    )
    a, rate = (np.ravel(item) for item in _series(trajectory, 'universe.a', 'universe.a_d1'))
    curvature = 1 - 0.3 - 0.01 - 0.8
    constraint = rate**2 - (0.3 / a + 0.01 / a**2 + 0.8 * a**2 + curvature)
    assert np.abs(constraint).max() < 1e-8
    assert rate[-1] > rate[len(rate) // 2]


def test_flrw_cosmology_rejects_a_universe_that_cannot_expand():
    """Test that a closed universe beyond its turning point is rejected."""
    with pytest.raises(ValidationError, match='negative squared expansion rate'):
        flrw_cosmology(3.0, omega_matter=3.0, omega_lambda=0.0)


@pytest.fixture(scope='module')
def collapse():
    """Return the times and radii of a collapsing ball of eight shells."""
    trajectory = _run(oppenheimer_snyder_collapse(8), 34.5, 0.5, ACCURATE)
    (radii,) = _series(trajectory, 'ball.R')
    return np.asarray(trajectory.times), radii


def test_dust_ball_collapses_along_the_oppenheimer_snyder_cycloid(collapse):
    """Test that every shell keeps `R = r a(t)` on the cycloid `a = (1 + cos η) / 2`, `t = (η + sin η) / (2 √k)`."""
    times, radii = collapse
    comoving = 10.0 * np.arange(1, 9) / 8
    k = 2 * 1.0 / 10.0**3
    angles = [brentq(lambda eta, t=t: (eta + math.sin(eta)) / (2 * math.sqrt(k)) - t, 0.0, math.pi) for t in times]
    scale = (1 + np.cos(angles)) / 2
    assert radii / comoving == pytest.approx(np.repeat(scale[:, None], 8, axis=1), rel=1e-7)


def test_dust_ball_surface_crosses_its_schwarzschild_radius(collapse):
    """Test that the surface falls inside `2 M` at the time the cycloid predicts, before the ball reaches zero size."""
    times, radii = collapse
    k = 2 * 1.0 / 10.0**3
    eta = math.acos(2 * 2.0 / 10.0 - 1)
    crossing = (eta + math.sin(eta)) / (2 * math.sqrt(k))
    surface = radii[:, -1]
    index = int(np.flatnonzero(surface < 2.0)[0])
    assert times[index - 1] < crossing <= times[index]
    assert times[-1] < math.pi / (2 * math.sqrt(k))


def test_oppenheimer_snyder_collapse_rejects_a_ball_inside_its_horizon():
    """Test that a ball that starts inside its Schwarzschild radius is rejected."""
    with pytest.raises(ValidationError, match='inside the Schwarzschild radius'):
        oppenheimer_snyder_collapse(radius=1.5)
