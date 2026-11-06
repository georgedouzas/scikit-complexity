"""Test the `_geodesics` module of `skcomplex.systems.physics.celestial_mechanics`."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.systems import create_simulation
from skcomplex.systems.physics.celestial_mechanics import (
    KerrSpace,
    SchwarzschildSpace,
    kerr_frame_dragging,
    light_bending,
    schwarzschild_precession,
)

from ..conftest import _run, _series

GEODESIC = ScipyODESolver(method='DOP853', rtol=1e-11, atol=1e-11)


def _norm(mass: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return `g_(mu nu) u^mu u^nu` of four-velocities in the Schwarzschild metric in the equatorial plane."""
    lapse = 1 - 2 * mass / x[:, 1]
    return -lapse * u[:, 0] ** 2 + u[:, 1] ** 2 / lapse + x[:, 1] ** 2 * u[:, 3] ** 2


def _periapsis_angles(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return the angles where the radial velocity turns from negative to positive, interpolated linearly."""
    radial, angle = u[:, 1], x[:, 3]
    index = np.flatnonzero((radial[:-1] < 0) & (radial[1:] >= 0))
    fraction = -radial[index] / (radial[index + 1] - radial[index])
    return angle[index] + fraction * (angle[index + 1] - angle[index])


def test_perihelion_advances_by_the_einstein_precession():
    """Test that a wide orbit advances its periapsis by `6 π M / (a (1 - e²))` per turn and stays timelike."""
    semi_major_axis, eccentricity = 1000.0, 0.5
    period = 2 * math.pi * semi_major_axis**1.5
    system = schwarzschild_precession(semi_major_axis, eccentricity)
    x, u = _series(_run(system, 2.2 * period, period / 4000, GEODESIC), 'particle.x', 'particle.x_d1')
    angles = _periapsis_angles(x, u)
    expected = 6 * math.pi / (semi_major_axis * (1 - eccentricity**2))
    assert len(angles) == 2
    assert np.diff(np.concatenate([[0.0], angles])) - 2 * math.pi == pytest.approx(expected, rel=0.02)
    assert _norm(1.0, x, u) == pytest.approx(-1.0, abs=1e-8)
    assert x[:, 2] == pytest.approx(math.pi / 2)


def test_circular_orbit_keeps_its_radius():
    """Test that an orbit without eccentricity keeps its radius."""
    x, _ = _series(_run(schwarzschild_precession(20.0, 0.0), 2000.0, 20.0, GEODESIC), 'particle.x', 'particle.x_d1')
    assert x[:, 1] == pytest.approx(20.0, rel=1e-7)


def test_light_ray_bends_by_four_masses_over_the_impact_parameter():
    """Test that the ray turns by the exact deflection of a null geodesic, `4 M / b` to first order, and stays null."""
    impact, distance = 100.0, 1e5
    system = light_bending(impact, distance)
    x, u = _series(_run(system, 2 * distance, 50.0, GEODESIC), 'photon.x', 'photon.x_d1')
    vx = u[:, 1] * np.cos(x[:, 3]) - x[:, 1] * np.sin(x[:, 3]) * u[:, 3]
    vy = u[:, 1] * np.sin(x[:, 3]) + x[:, 1] * np.cos(x[:, 3]) * u[:, 3]
    deflection = -math.atan2(vy[-1], vx[-1])
    assert deflection == pytest.approx(4 / impact + 15 * math.pi / (4 * impact**2), rel=0.01)
    assert _norm(1.0, x, u) == pytest.approx(0.0, abs=1e-8)
    turning = brentq(lambda inverse: 1 / impact**2 - inverse**2 + 2 * inverse**3, 0.0, 1 / impact * 1.1)

    def integrand(root: float) -> float:
        """Return the turning rate of the ray at `1/r = turning (1 - root²)`, regular at the closest approach."""
        inverse = turning * (1 - root**2)
        return 2 * turning * root / math.sqrt(1 / impact**2 - inverse**2 + 2 * inverse**3)

    exact = 2 * quad(integrand, 0.0, 1.0)[0] - math.pi
    assert exact == pytest.approx(4 / impact + 15 * math.pi / (4 * impact**2), rel=1e-2)
    assert deflection == pytest.approx(exact, rel=2e-3)


def test_zero_angular_momentum_particle_turns_with_the_frames():
    """Test that a particle without angular momentum turns at `dφ/dt = 2 M a / (r³ + a² r + 2 M a²)` as it falls."""
    spin = 0.9
    x, u = _series(_run(kerr_frame_dragging(10.0, spin=spin), 30.0, 0.5, GEODESIC), 'particle.x', 'particle.x_d1')
    r = x[:, 1]
    assert u[:, 3] / u[:, 0] == pytest.approx(2 * spin * 1.0 / (r**3 + spin**2 * r + 2 * spin**2), rel=1e-7)
    assert np.all(np.diff(r) < 0)
    assert np.all(np.diff(x[:, 3]) > 0)
    assert x[-1, 3] > 0.1


def test_static_mass_does_not_drag_a_falling_particle():
    """Test that a particle without angular momentum falls straight into a mass without spin."""
    x, _ = _series(_run(kerr_frame_dragging(10.0, spin=0.0), 30.0, 0.5, GEODESIC), 'particle.x', 'particle.x_d1')
    assert x[:, 3] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ('factory', 'kwargs', 'message'),
    [
        (schwarzschild_precession, {'eccentricity': 1.0}, 'bound orbit'),
        (schwarzschild_precession, {'semi_major_axis': 10.0}, 'innermost stable'),
        (light_bending, {'impact_parameter': 5.0}, 'captures'),
        (light_bending, {'distance': 10.0}, 'not larger'),
        (kerr_frame_dragging, {'radius': 1.5}, 'ergosphere'),
    ],
)
def test_geodesic_systems_reject_invalid_orbits(factory, kwargs, message):
    """Test that bound orbits, captured rays, and starts inside the ergosphere are rejected."""
    with pytest.raises(ValidationError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize(
    'space',
    [SchwarzschildSpace(mass=-1.0), KerrSpace(mass=1.0, spin=2.0)],
)
def test_spacetimes_reject_invalid_masses_and_spins(space):
    """Test that a mass that is not positive, or a spin above the mass, is rejected."""
    with pytest.raises(ValidationError, match='Give a positive mass'):
        space.prepare()


def test_registered_defaults_run():
    """Test that the precession runs with its registered settings."""
    simulation = create_simulation('schwarzschild_precession').prepare().run()
    assert np.asarray(simulation.trajectory_.values('particle.x')).shape[1] == 4
