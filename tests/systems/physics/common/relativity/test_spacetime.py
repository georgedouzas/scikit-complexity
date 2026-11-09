"""Test the `_spacetime` module of `skcomplex.systems.physics.common.relativity`."""

import math

import numpy as np
import pytest
from sage.all__sagemath_symbolics import SR, diagonal_matrix, function, matrix, sin
from sklearn.base import clone

from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.core.errors import SpaceError, ValidationError
from skcomplex.systems.general.elements import PointElement
from skcomplex.systems.physics.common.relativity import GeodesicInteraction, SpacetimeSpace

pytest.importorskip('einsteinpy')

t, r, theta, phi, M = SR.var('t r theta phi M')


def _schwarzschild() -> SpacetimeSpace:
    """Return the Schwarzschild space of a symbolic mass."""
    lapse = 1 - 2 * M / r
    metric = diagonal_matrix(SR, [-lapse, 1 / lapse, r**2, r**2 * sin(theta) ** 2])
    return SpacetimeSpace(metric, (t, r, theta, phi))


@pytest.mark.parametrize(
    ('space', 'message'),
    [
        (SpacetimeSpace(), 'not variables'),
        (SpacetimeSpace(matrix(SR, [[1, 0], [0, 1]]), ('r', 'r')), 'repeat'),
        (SpacetimeSpace(matrix(SR, [[1, 0], [0, 1]]), ('r',)), 'one row per coordinate'),
        (SpacetimeSpace(matrix(SR, [[1, r], [0, 1]]), ('r', 'phi')), 'not symmetric'),
        (SpacetimeSpace(matrix(SR, [[1, 0], [0, 0]]), ('r', 'phi')), 'degenerate'),
        (SpacetimeSpace(None, ('r', 'phi')), 'no metric'),
    ],
)
def test_pseudo_riemannian_space_rejects_invalid_metrics(space, message):
    """Test that an invalid metric is built without error and rejected by its check."""
    with pytest.raises(ValidationError, match=message):
        space.prepare()


def test_pseudo_riemannian_space_is_an_unbounded_continuous_space():
    """Test that the space has one dimension per coordinate, no sites, and no walls."""
    space = SpacetimeSpace([[1, 0], [0, r**2]], ('r', 'phi')).prepare()
    assert isinstance(space, ContinuousSpace)
    assert (space.n_dim_, space.size_, space.shape_, space.periodic_, space.extent_) == (2, None, (), False, ())
    assert space.coordinates_ == (r, phi)
    assert clone(space).get_params()['coordinates'] == ('r', 'phi')


def test_sphere_has_constant_positive_curvature():
    """Test that the unit sphere has `Γ^θ_φφ = -sin θ cos θ`, `Γ^φ_θφ = cot θ`, and the Ricci scalar 2."""
    space = SpacetimeSpace(matrix(SR, [[1, 0], [0, sin(theta) ** 2]]), (theta, phi)).prepare()
    assert bool((space.christoffel_[0][1][1] + sin(theta) * SR('cos(theta)')).simplify_full() == 0)
    assert bool((space.christoffel_[1][0][1] - SR('cot(theta)')).simplify_full() == 0)
    assert space.ricci_scalar_ == 2
    assert bool((space.riemann_[0][1][0][1] - sin(theta) ** 2).simplify_full() == 0)


def test_schwarzschild_space_is_ricci_flat_with_known_connection():
    """Test that the Schwarzschild metric has a zero Ricci and Einstein tensor and `Γ^r_tt = M (r - 2M) / r³`."""
    space = _schwarzschild().prepare()
    assert all(bool(SR(entry).simplify_full() == 0) for entry in space.ricci_.list())
    assert all(bool(SR(entry).simplify_full() == 0) for entry in space.einstein_.list())
    assert bool((space.christoffel_[1][0][0] - M * (r - 2 * M) / r**3).simplify_full() == 0)
    assert bool((space.christoffel_[0][0][1] - M / (r * (r - 2 * M))).simplify_full() == 0)


def test_space_without_curvature_derives_only_the_connection():
    """Test that `curvature=False` leaves the curvature tensors out."""
    space = _schwarzschild().set_params(curvature=False).prepare()
    assert space.christoffel_[2][1][2] == 1 / r
    assert (space.riemann_, space.ricci_, space.ricci_scalar_, space.einstein_) == (None, None, None, None)


def test_ansatz_functions_survive_names_sympy_reserves():
    """Test that functions of an ansatz named like SymPy constants, such as `E`, come back as the same functions."""
    E = function('E')(r)
    space = SpacetimeSpace(diagonal_matrix(SR, [1 + 2 * E, r**2]), (r, phi)).prepare()
    assert bool((space.christoffel_[0][0][0] - E.diff(r) / (1 + 2 * E)).simplify_full() == 0)


def test_friedmann_universe_has_the_einstein_tensor_of_the_friedmann_equations():
    """Test that the flat Robertson-Walker ansatz has `G^t_t = -3 (a'/a)²` and `G^r_r = -(2 a a'' + a'²) / a²`."""
    a = function('a')(t)
    metric = diagonal_matrix(SR, [-1, a**2, a**2 * r**2, a**2 * r**2 * sin(theta) ** 2])
    space = SpacetimeSpace(metric, (t, r, theta, phi)).prepare()
    mixed = space.inverse_metric_ * space.einstein_
    assert bool((mixed[0, 0] + 3 * a.diff(t) ** 2 / a**2).simplify_full() == 0)
    assert bool((mixed[1, 1] + (2 * a * a.diff(t, 2) + a.diff(t) ** 2) / a**2).simplify_full() == 0)


def test_prepared_space_derives_again_only_when_its_source_changes():
    """Test that preparing again keeps the derived tensors of an equal metric and derives them for a changed one."""
    space = _schwarzschild().set_params(curvature=False).prepare()
    christoffel = space.christoffel_
    assert space.prepare().christoffel_ is christoffel
    assert space.set_params(curvature=True).prepare().christoffel_ is not christoffel
    assert space.ricci_ is not None
    plane = SpacetimeSpace([[1, 0], [0, r**2]], ('r', 'phi')).prepare()
    assert plane.set_params(metric=[[1, 0], [0, 4 * r**2]]).prepare().christoffel_[0][1][1] == -4 * r


ACCURATE = ScipyODESolver(method='DOP853', rtol=1e-11, atol=1e-11)

r, phi, theta = SR.var('r phi theta')


def _run(space: SpacetimeSpace, x: list[float], x_d1: list[float], until: float) -> np.ndarray:
    """Return the positions and tangents of a point that follows the geodesics of a space."""
    point = PointElement('point', x=x, x_d1=x_d1)
    geodesic = GeodesicInteraction('geodesic', target='point', space=space)
    system = CompositeSystem(elements=[point], interactions=[geodesic], space=space)
    trajectory = simulate(system, until=until, dt=until / 50, solver=ACCURATE)
    return np.hstack([trajectory.values('point.x'), trajectory.values('point.x_d1')])


def test_geodesic_of_the_polar_plane_is_a_straight_line():
    """Test that a geodesic of the plane in polar coordinates is the straight line `x = 1` at unit speed."""
    plane = SpacetimeSpace(matrix(SR, [[1, 0], [0, r**2]]), (r, phi))
    states = _run(plane, [1.0, 0.0], [0.0, 1.0], 3.0)
    x, y = states[:, 0] * np.cos(states[:, 1]), states[:, 0] * np.sin(states[:, 1])
    assert x == pytest.approx(1.0, abs=1e-8)
    assert np.diff(y) == pytest.approx(3.0 / 50, abs=1e-8)


def test_geodesic_of_the_sphere_is_a_great_circle():
    """Test that a geodesic of the unit sphere through the pole stays on a meridian and turns at unit speed."""
    sphere = SpacetimeSpace(matrix(SR, [[1, 0], [0, sin(theta) ** 2]]), (theta, phi))
    states = _run(sphere, [0.5, 0.3], [1.0, 0.0], 2.0)
    assert states[-1, 0] == pytest.approx(2.5, abs=1e-8)
    assert states[:, 1] == pytest.approx(0.3, abs=1e-10)


def test_geodesic_keeps_the_norm_of_its_tangent():
    """Test that a null tangent of two-dimensional Minkowski space in polar time stays null."""
    t = SR.var('t')
    space = SpacetimeSpace(matrix(SR, [[-(r**2), 0], [0, 1]]), (t, r))
    states = _run(space, [0.0, 1.0], [1.0, 1.0], 1.0)
    norm = -(states[:, 1] ** 2) * states[:, 2] ** 2 + states[:, 3] ** 2
    assert norm == pytest.approx(0.0, abs=1e-8)
    assert states[-1, 1] > 1.0


def test_geodesic_interaction_returns_one_rate_law_of_the_position():
    """Test that the law is `x''^mu = -Γ^mu_(alpha beta) x'^alpha x'^beta` on the vector position."""
    plane = SpacetimeSpace(matrix(SR, [[1, 0], [0, r**2]]), (r, phi))
    geodesic = GeodesicInteraction('geodesic', target='point', space=plane).prepare()
    assert GeodesicInteraction.law_classes == (RateLaw,)
    assert geodesic.space_.christoffel_[0][1][1] == -r
    assert math.isclose(float(geodesic.space_.christoffel_[1][0][1].subs(r=2.0)), 0.5)


@pytest.mark.parametrize(
    ('geodesic', 'error', 'message'),
    [
        (GeodesicInteraction('geodesic', target='point', state='1x'), ValidationError, 'name of a state'),
        (GeodesicInteraction('geodesic', target='point', space=ContinuousSpace(n_dim=2)), SpaceError, 'Spacetime'),
        (GeodesicInteraction('geodesic', target='point'), SpaceError, 'Spacetime'),
    ],
)
def test_geodesic_interaction_rejects_invalid_arguments(geodesic, error, message):
    """Test that a state that is no name, or a space that has no metric, is rejected by `prepare`."""
    with pytest.raises(error, match=message):
        geodesic.prepare()


def test_geodesic_interaction_rejects_a_position_of_another_dimension():
    """Test that a position of three components on a space of two coordinates is rejected."""
    plane = SpacetimeSpace(matrix(SR, [[1, 0], [0, r**2]]), (r, phi))
    point = PointElement('point', x=[1.0, 0.0, 0.0], x_d1=0.0)
    geodesic = GeodesicInteraction('geodesic', target='point', space=plane)
    system = CompositeSystem(elements=[point], interactions=[geodesic], space=ContinuousSpace(n_dim=3))
    with pytest.raises(ValidationError, match='3 components for the 2 coordinates'):
        simulate(system, until=0, dt=0.1)
