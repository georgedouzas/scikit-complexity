"""Test the exact Jacobian of drifts that read reductions, and its sparsity pattern."""

import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin, tanh

from skcomplex.core.base import ValidationError
from skcomplex.core.compile import compile_system, mesh_laplacian
from skcomplex.core.components import Element, Field
from skcomplex.core.laws import Rate, count, laplacian, maximum, mean, minimum, total, variance, where
from skcomplex.core.solvers import SplitOperator
from skcomplex.core.space import LatticeSpace, Mesh
from skcomplex.core.system import System
from skcomplex.models.biology.ecology import FisherKPP
from skcomplex.models.chemistry.pattern_formation import gray_scott
from skcomplex.models.physics.fluids import kuramoto_sivashinsky
from skcomplex.models.physics.heat import heat_diffusion, heat_l_shape
from skcomplex.models.physics.synchronization import nonlocal_ring_chimera

RNG = np.random.default_rng(1)
PAYOFF = RNG.normal(size=(20, 20))


def _finite_differences(problem, x, p, step=1e-6):
    """Return the central finite-difference Jacobian of a drift."""
    columns = []
    for index in range(x.size):
        shift = np.zeros(x.size)
        shift[index] = step
        columns.append((problem.drift(0.0, x + shift, p) - problem.drift(0.0, x - shift, p)) / (2 * step))
    return np.stack(columns, axis=1)


@pytest.mark.parametrize(
    'laws',
    [
        lambda s: [Rate(s.x, -s.x + tanh(total(s.x)) + mean(s.x**2))],
        lambda s: [
            Rate(s.x, s.x * (total(s.x, weights=PAYOFF) - total(s.x * total(s.x, weights=PAYOFF)))),
            Rate(
                s.y,
                maximum(s.x * s.y)
                + variance(s.y)
                - minimum(sin(s.x))
                + mean(s.y, where=s.x > 0) * count(where=s.y > 0),
            ),
        ],
        lambda s: [Rate(s.x, where(s.x > 0, total(s.y**2), -s.x)), Rate(s.y, -s.y)],
    ],
    ids=['tanh of total', 'every kind', 'condition'],
)
def test_jacobian_against_finite_differences(laws):
    """Test that the exact Jacobian equals central finite differences and that its pattern covers every entry."""
    Member = Element.define('Member', states={'x': 0.0, 'y': 0.0}, laws=laws)
    system = System(components=[Member.group('group', 20, x=RNG.normal(size=20), y=RNG.normal(size=20))])
    problem = compile_system(system)
    x, p = problem.initial_state, problem.initial_parameters
    exact = problem.jacobian(0.0, x, p).toarray()
    estimate = _finite_differences(problem, x, p)
    np.testing.assert_allclose(exact, estimate, atol=1e-6)
    pattern = problem.jacobian.pattern().toarray()
    assert pattern[np.abs(estimate) > 1e-9].all()


def test_jacobian_of_pair_sums_is_not_derived():
    """Test that the exact Jacobian of a pair interaction raises an error that names the interaction."""
    problem = compile_system(nonlocal_ring_chimera(10, R=2))
    with pytest.raises(ValidationError, match=r'coupling.*finite differences'):
        problem.jacobian(0.0, problem.initial_state, problem.initial_parameters)


def test_pattern_is_omitted_above_the_limit():
    """Test that the pattern of a dense block above the size limit is omitted."""
    Member = Element.define('Wide', states={'x': 0.0}, laws=lambda s: [Rate(s.x, total(s.x))])
    problem = compile_system(System(components=[Member.group('group', 3000)]))
    assert problem.jacobian.pattern() is None


ReactiveHeat = Element.define(
    'ReactiveHeat',
    states={'u': 0.0},
    parameters={'D': 1.0, 'r': 1.0},
    laws=lambda s: [Rate(s.u, s.D * laplacian(s.u) + s.r * s.u * (1 - s.u**2))],
)


def _reactive_mesh(refine=2):
    """Return heat with a cubic reaction on the L-shaped mesh with fixed boundary temperatures."""
    plate = Field(
        'plate',
        ReactiveHeat,
        Mesh.l_shape(refine),
        boundary={'u': ('fixed', 0.5)},
        discretization='finite-element',
    )
    return System(components=[plate])


def _fisher_line(size, boundary):
    """Return Fisher-KPP on a line with an edge condition."""
    space = LatticeSpace(size, spacing=0.25)
    return System(components=[Field('line', FisherKPP, space, boundary={'u': boundary})])


@pytest.mark.parametrize(('boundary', 'entries'), [('zero-flux', 2998), ('periodic', 3000)])
def test_stencil_pattern(boundary, entries):
    """Test that the three-point Laplacian gives a tridiagonal pattern, closed into a ring by periodic edges."""
    pattern = compile_system(_fisher_line(1000, boundary)).jacobian.pattern()
    assert pattern.nnz == entries
    rows, columns = pattern.nonzero()
    assert set(np.abs(rows - columns)) <= ({0, 1, 999} if boundary == 'periodic' else {0, 1})


def test_mesh_pattern():
    """Test that the mesh pattern is the stiffness pattern with the diagonal on free nodes and empty fixed rows."""
    problem = compile_system(_reactive_mesh())
    pattern = problem.jacobian.pattern().toarray()
    stiffness = mesh_laplacian(Mesh.l_shape(2)).toarray() != 0
    free = np.setdiff1d(np.arange(problem.initial_state.size), problem.constrained)
    assert problem.constrained.size
    assert not pattern[problem.constrained].any()
    expected = stiffness | np.eye(len(stiffness), dtype=bool)
    np.testing.assert_array_equal(pattern[free], expected[free])


@pytest.mark.parametrize('system', [heat_diffusion((12, 12)), heat_l_shape(2)], ids=['lattice', 'mesh'])
def test_pattern_agrees_with_split_operator(system):
    """Test that the pattern holds every entry of the sparse linear part of linear lattice and mesh fields."""
    problem = compile_system(system)
    operator = SplitOperator.build(problem, problem.initial_parameters).sparse
    pattern = problem.jacobian.pattern().toarray()
    assert pattern[operator.toarray() != 0].all()


@pytest.mark.parametrize(
    ('system', 'block'),
    [(nonlocal_ring_chimera(12, R=2), 12), (kuramoto_sivashinsky(16), 16)],
    ids=['pairs', 'spectral'],
)
def test_run_time_couplings_are_dense(system, block):
    """Test that pair interactions and spectral operators give the dense block of their group or field."""
    pattern = compile_system(system).jacobian.pattern().toarray()
    assert pattern[:block, :block].all()


def _relative_error(problem, rng):
    """Return the relative Frobenius error of the exact Jacobian against central finite differences."""
    x = problem.initial_state + 0.1 * rng.random(problem.initial_state.size)
    p = problem.initial_parameters
    exact = problem.jacobian(0.0, x, p).toarray()
    estimate = _finite_differences(problem, x, p)
    return np.linalg.norm(exact - estimate) / np.linalg.norm(estimate)


def test_exact_jacobian_of_gray_scott():
    """Test that the exact Jacobian of Gray-Scott on a 64 by 64 lattice agrees with central finite differences."""
    problem = compile_system(gray_scott((64, 64)), backend='numpy')
    assert _relative_error(problem, np.random.default_rng(0)) < 1e-6


def test_exact_jacobian_of_mesh_with_reaction():
    """Test that the exact Jacobian of heat with a reaction on a mesh matches finite differences, fixed rows empty."""
    problem = compile_system(_reactive_mesh())
    assert _relative_error(problem, np.random.default_rng(0)) < 1e-6
    jacobian = problem.jacobian(0.0, problem.initial_state, problem.initial_parameters)
    assert not jacobian[problem.constrained].nnz


def test_exact_jacobian_reads_the_parameters_it_receives():
    """Test that the exact Jacobian is evaluated at the parameters of the call."""
    problem = compile_system(_fisher_line(20, 'zero-flux'))
    x, p = problem.initial_state, problem.initial_parameters.copy()
    first = problem.jacobian(0.0, x, p).toarray()
    p[problem.parameters.entry('line.D').indices] *= 2
    second = problem.jacobian(0.0, x, p).toarray()
    off = ~np.eye(20, dtype=bool)
    np.testing.assert_allclose(second[off], 2 * first[off])


def _callable_rate(t, values):
    """Return the decay of the line."""
    return -values['line.u']


def test_exact_jacobian_of_callable_law_raises():
    """Test that the exact Jacobian of a callable law raises an error naming the law and the automatic option."""
    Decay = Element.define('Decay', states={'u': 0.0}, laws=lambda s: [Rate(s.u, _callable_rate)])
    problem = compile_system(System(components=[Field('line', Decay, LatticeSpace(10))]))
    with pytest.raises(ValidationError, match=r"line.*jacobian='auto'"):
        problem.jacobian.validate()
    assert problem.jacobian.pattern().toarray().all()
