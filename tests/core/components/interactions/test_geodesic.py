"""Test the `_geodesic` module of `skcomplex.core.components.interactions`."""

import math

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, matrix, sin

from skcomplex.core.components.elements import PointElement
from skcomplex.core.components.interactions import GeodesicInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import ContinuousSpace, PseudoRiemannianSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import SpaceError, ValidationError

ACCURATE = ScipyODESolver(method='DOP853', rtol=1e-11, atol=1e-11)

r, phi, theta = SR.var('r phi theta')


def _run(space: PseudoRiemannianSpace, x: list[float], x_d1: list[float], until: float) -> np.ndarray:
    """Return the positions and tangents of a point that follows the geodesics of a space."""
    point = PointElement('point', x=x, x_d1=x_d1)
    geodesic = GeodesicInteraction('geodesic', target='point', space=space)
    system = CompositeSystem(elements=[point], interactions=[geodesic], space=space)
    trajectory = simulate(system, until=until, dt=until / 50, solver=ACCURATE)
    return np.hstack([trajectory.values('point.x'), trajectory.values('point.x_d1')])


def test_geodesic_of_the_polar_plane_is_a_straight_line():
    """Test that a geodesic of the plane in polar coordinates is the straight line `x = 1` at unit speed."""
    plane = PseudoRiemannianSpace(matrix(SR, [[1, 0], [0, r**2]]), (r, phi))
    states = _run(plane, [1.0, 0.0], [0.0, 1.0], 3.0)
    x, y = states[:, 0] * np.cos(states[:, 1]), states[:, 0] * np.sin(states[:, 1])
    assert x == pytest.approx(1.0, abs=1e-8)
    assert np.diff(y) == pytest.approx(3.0 / 50, abs=1e-8)


def test_geodesic_of_the_sphere_is_a_great_circle():
    """Test that a geodesic of the unit sphere through the pole stays on a meridian and turns at unit speed."""
    sphere = PseudoRiemannianSpace(matrix(SR, [[1, 0], [0, sin(theta) ** 2]]), (theta, phi))
    states = _run(sphere, [0.5, 0.3], [1.0, 0.0], 2.0)
    assert states[-1, 0] == pytest.approx(2.5, abs=1e-8)
    assert states[:, 1] == pytest.approx(0.3, abs=1e-10)


def test_geodesic_keeps_the_norm_of_its_tangent():
    """Test that a null tangent of two-dimensional Minkowski space in polar time stays null."""
    t = SR.var('t')
    space = PseudoRiemannianSpace(matrix(SR, [[-(r**2), 0], [0, 1]]), (t, r))
    states = _run(space, [0.0, 1.0], [1.0, 1.0], 1.0)
    norm = -(states[:, 1] ** 2) * states[:, 2] ** 2 + states[:, 3] ** 2
    assert norm == pytest.approx(0.0, abs=1e-8)
    assert states[-1, 1] > 1.0


def test_geodesic_interaction_returns_one_rate_law_of_the_position():
    """Test that the law is `x''^mu = -Γ^mu_(alpha beta) x'^alpha x'^beta` on the vector position."""
    plane = PseudoRiemannianSpace(matrix(SR, [[1, 0], [0, r**2]]), (r, phi))
    geodesic = GeodesicInteraction('geodesic', target='point', space=plane).prepare()
    assert GeodesicInteraction.law_classes == (RateLaw,)
    assert geodesic.space_.christoffel_[0][1][1] == -r
    assert math.isclose(float(geodesic.space_.christoffel_[1][0][1].subs(r=2.0)), 0.5)


@pytest.mark.parametrize(
    ('geodesic', 'error', 'message'),
    [
        (GeodesicInteraction('geodesic', target='point', state='1x'), ValidationError, 'name of a state'),
        (GeodesicInteraction('geodesic', target='point', space=ContinuousSpace(n_dim=2)), SpaceError, 'Pseudo'),
        (GeodesicInteraction('geodesic', target='point'), SpaceError, 'Pseudo'),
    ],
)
def test_geodesic_interaction_rejects_invalid_arguments(geodesic, error, message):
    """Test that a state that is no name, or a space that has no metric, is rejected by `prepare`."""
    with pytest.raises(error, match=message):
        geodesic.prepare()


def test_geodesic_interaction_rejects_a_position_of_another_dimension():
    """Test that a position of three components on a space of two coordinates is rejected."""
    plane = PseudoRiemannianSpace(matrix(SR, [[1, 0], [0, r**2]]), (r, phi))
    point = PointElement('point', x=[1.0, 0.0, 0.0], x_d1=0.0)
    geodesic = GeodesicInteraction('geodesic', target='point', space=plane)
    system = CompositeSystem(elements=[point], interactions=[geodesic], space=ContinuousSpace(n_dim=3))
    with pytest.raises(ValidationError, match='3 components for the 2 coordinates'):
        simulate(system, until=0, dt=0.1)
