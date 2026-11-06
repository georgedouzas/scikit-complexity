"""Test the `_pseudo_riemannian` module of `skcomplex.core.components.spaces`."""

import pytest
from sage.all__sagemath_symbolics import SR, diagonal_matrix, function, matrix, sin
from sklearn.base import clone

from skcomplex.core.components.spaces import ContinuousSpace, PseudoRiemannianSpace
from skcomplex.core.errors import ValidationError

t, r, theta, phi, M = SR.var('t r theta phi M')


def _schwarzschild() -> PseudoRiemannianSpace:
    """Return the Schwarzschild space of a symbolic mass."""
    lapse = 1 - 2 * M / r
    metric = diagonal_matrix(SR, [-lapse, 1 / lapse, r**2, r**2 * sin(theta) ** 2])
    return PseudoRiemannianSpace(metric, (t, r, theta, phi))


@pytest.mark.parametrize(
    ('space', 'message'),
    [
        (PseudoRiemannianSpace(), 'not variables'),
        (PseudoRiemannianSpace(matrix(SR, [[1, 0], [0, 1]]), ('r', 'r')), 'repeat'),
        (PseudoRiemannianSpace(matrix(SR, [[1, 0], [0, 1]]), ('r',)), 'one row per coordinate'),
        (PseudoRiemannianSpace(matrix(SR, [[1, r], [0, 1]]), ('r', 'phi')), 'not symmetric'),
        (PseudoRiemannianSpace(matrix(SR, [[1, 0], [0, 0]]), ('r', 'phi')), 'degenerate'),
        (PseudoRiemannianSpace(None, ('r', 'phi')), 'no metric'),
    ],
)
def test_pseudo_riemannian_space_rejects_invalid_metrics(space, message):
    """Test that an invalid metric is built without error and rejected by its check."""
    with pytest.raises(ValidationError, match=message):
        space.prepare()


def test_pseudo_riemannian_space_is_an_unbounded_continuous_space():
    """Test that the space has one dimension per coordinate, no sites, and no walls."""
    space = PseudoRiemannianSpace([[1, 0], [0, r**2]], ('r', 'phi')).prepare()
    assert isinstance(space, ContinuousSpace)
    assert (space.n_dim_, space.size_, space.shape_, space.periodic_, space.extent_) == (2, None, (), False, ())
    assert space.coordinates_ == (r, phi)
    assert clone(space).get_params()['coordinates'] == ('r', 'phi')


def test_sphere_has_constant_positive_curvature():
    """Test that the unit sphere has `Γ^θ_φφ = -sin θ cos θ`, `Γ^φ_θφ = cot θ`, and the Ricci scalar 2."""
    space = PseudoRiemannianSpace(matrix(SR, [[1, 0], [0, sin(theta) ** 2]]), (theta, phi)).prepare()
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
    space = PseudoRiemannianSpace(diagonal_matrix(SR, [1 + 2 * E, r**2]), (r, phi)).prepare()
    assert bool((space.christoffel_[0][0][0] - E.diff(r) / (1 + 2 * E)).simplify_full() == 0)


def test_friedmann_universe_has_the_einstein_tensor_of_the_friedmann_equations():
    """Test that the flat Robertson-Walker ansatz has `G^t_t = -3 (a'/a)²` and `G^r_r = -(2 a a'' + a'²) / a²`."""
    a = function('a')(t)
    metric = diagonal_matrix(SR, [-1, a**2, a**2 * r**2, a**2 * r**2 * sin(theta) ** 2])
    space = PseudoRiemannianSpace(metric, (t, r, theta, phi)).prepare()
    mixed = space.inverse_metric_ * space.einstein_
    assert bool((mixed[0, 0] + 3 * a.diff(t) ** 2 / a**2).simplify_full() == 0)
    assert bool((mixed[1, 1] + (2 * a * a.diff(t, 2) + a.diff(t) ** 2) / a**2).simplify_full() == 0)
