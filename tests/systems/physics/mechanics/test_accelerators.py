"""Test the `_accelerators` module of `skcomplex.systems.physics.mechanics`."""

import math

import numpy as np
import pytest

from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.mechanics import BeamBeamMapSystem, FrictionOscillatorSystem


def _beam_beam_orbit(x, y, nu, c, until):
    """Return the positions and momenta of an orbit of the beam-beam map."""
    trajectory = simulate(BeamBeamMapSystem(x, y, nu=nu, c=c), until=until)
    return np.ravel(trajectory.values('particle.x')), np.ravel(trajectory.values('particle.y'))


def test_beam_beam_map_preserves_area():
    """Test that one turn of the beam-beam map keeps the area of a small triangle of initial points."""
    corners = [(0.7, 0.2), (0.7 + 1e-5, 0.2), (0.7, 0.2 + 1e-5)]
    images = np.array([[value[1] for value in _beam_beam_orbit(x, y, 0.3, 0.05, 1)] for x, y in corners])
    sides = images[1:] - images[0]
    assert abs(sides[0, 0] * sides[1, 1] - sides[0, 1] * sides[1, 0]) == pytest.approx(1e-10, rel=1e-4)


@pytest.mark.parametrize(('nu', 'stable'), [(0.3, True), (0.9, False)])
def test_beam_beam_map_origin_is_stable_when_the_trace_lies_within_two(nu, stable):
    """Test that orbits near the origin stay close for `|2 a - 4 pi c b| < 2` and leave it otherwise."""
    c = 0.05
    trace = 2 * math.cos(math.pi * nu) - 4 * math.pi * c * math.sin(math.pi * nu)
    assert (abs(trace) < 2) == stable
    x, y = _beam_beam_orbit(1e-4, 0.0, nu, c, 500)
    radius = np.hypot(x, y).max()
    assert radius < 1e-3 if stable else radius > 1e-2


def test_friction_oscillator_loses_a_fixed_amplitude_every_swing_and_sticks():
    """Test that every half swing ends `2 mu_k m g / k` closer and the mass sticks at the first turn inside the band."""
    trajectory = simulate(FrictionOscillatorSystem(), random_state=0)
    x, v = np.ravel(trajectory.values('mass.x')), np.ravel(trajectory.values('mass.v'))
    turns = x[np.flatnonzero(np.sign(v[1:]) != np.sign(v[:-1]))]
    expected = [(-1) ** n * (1.03 - 0.1 * n) for n in range(10)]
    np.testing.assert_allclose(turns[:10], expected, atol=1e-3)
    assert x[-1] == pytest.approx(0.03, abs=1e-3)
    assert np.ptp(x[-100:]) == 0


def test_friction_oscillator_stays_put_inside_the_static_band():
    """Test that a mass released inside the band `|k x| <= mu_s m g` never moves."""
    trajectory = simulate(FrictionOscillatorSystem(x=0.055), until=5.0, dt=0.1)
    assert np.ptp(np.ravel(trajectory.values('mass.x'))) == 0
