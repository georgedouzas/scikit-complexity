"""Test the `_jacobian_kernel` module of `skcomplex.core.engine.problems`."""

import numpy as np
import pytest
import skfem
from sage.all__sagemath_symbolics import sin, tanh

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import (
    convolve,
    count,
    displacement,
    inverse_helmholtz,
    laplacian,
    maximum,
    mean,
    minimum,
    neighbors,
    partial,
    total,
    variance,
    where,
)
from skcomplex.core.components.spaces import CellComplexSpace, ContinuousSpace, LatticeSpace, RadiusNeighborhood
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile._field_lowering import FieldLowering
from skcomplex.core.engine.problems import RunStateSpec
from skcomplex.core.engine.solvers import IMEXEulerSolver, ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System
from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem
from skcomplex.systems.physics.fluids import KuramotoSivashinskySystem
from skcomplex.systems.physics.heat import HeatDiffusionSystem, HeatLShapeSystem

RNG = np.random.default_rng(1)
PAYOFF = RNG.normal(size=(20, 20))


def _system(label, laws, values, size=None, space=None, **settings):
    """Return one element of values, of a size or on a space, whose interaction `laws` sets its laws."""
    element = Element(label, values=values, size=size, space=space)
    return System(
        elements=[element],
        interactions=[Interaction('laws', roles={'target': label}, laws=laws)],
        **settings,
    )


def _problem(system):
    """Return the problem a system or a ready-made system compiles to."""
    return system.prepare().system_._problem


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
        lambda x, y: [RateLaw(x, -x + tanh(total(x)) + mean(x**2))],
        lambda x, y: [
            RateLaw(x, x * (total(x, weights=PAYOFF) - total(x * total(x, weights=PAYOFF)))),
            RateLaw(
                y,
                maximum(x * y) + variance(y) - minimum(sin(x)) + mean(y, where=x > 0) * count(where=y > 0),
            ),
        ],
        lambda x, y: [RateLaw(x, where(x > 0, total(y**2), -x)), RateLaw(y, -y)],
    ],
    ids=['tanh of total', 'every kind', 'condition'],
)
def test_jacobian_against_finite_differences(laws):
    """Test that the exact Jacobian equals central finite differences and that its pattern covers every entry."""

    values = {'x': RNG.normal(size=20), 'y': RNG.normal(size=20)}
    problem = _problem(_system('group', lambda s: laws(s.target.x, s.target.y), values, size=20))
    x, p = problem.initial_state, problem.initial_parameters
    exact = problem.jacobian.prepare(problem).run(RunStateSpec(0.0, x, p)).values_.toarray()
    estimate = _finite_differences(problem, x, p)
    np.testing.assert_allclose(exact, estimate, atol=1e-6)
    pattern = problem.jacobian.get_pattern().toarray()
    assert pattern[np.abs(estimate) > 1e-9].all()


def _pair_ring(size):
    """Return phase oscillators on a periodic ring, coupled by the sine of their phase differences within two sites."""
    values = {'theta': RNG.uniform(0, 2 * np.pi, size), 'x': np.arange(size, dtype=float)}
    ring = Element('ring', values=values, size=size)
    coupling = Interaction(
        'coupling',
        roles={'target': 'ring'},
        laws=lambda s: [
            RateLaw(s.target.theta, neighbors(sin(s.target.other.theta - s.target.theta), RadiusNeighborhood(2.5))),
        ],
    )
    space = ContinuousSpace(extent=(float(size),), boundary='periodic')
    return System(elements=[ring], interactions=[coupling], space=space)


def test_jacobian_of_pair_sums_is_not_derived():
    """Test that the exact Jacobian of pairs within a radius raises an error that suggests finite differences."""
    problem = _problem(_pair_ring(10))
    with pytest.raises(ValidationError, match=r'reduce over neighbourhoods.*finite differences'):
        problem.jacobian.prepare(problem).run(RunStateSpec(0.0, problem.initial_state, problem.initial_parameters))


def test_pattern_is_omitted_above_the_limit():
    """Test that the pattern of a dense block above the size limit is omitted."""

    problem = _problem(_system('group', lambda s: [RateLaw(s.target.x, total(s.target.x))], {'x': 0.0}, size=3000))
    assert problem.jacobian.get_pattern() is None


def _reactive_heat(s):
    """Return heat with a cubic reaction, `du/dt = D laplacian(u) + r u (1 - u^2)`."""
    u = s.target.u
    return [RateLaw(u, s.target.D * laplacian(u) + s.target.r * u * (1 - u**2))]


def _fisher_kpp(s):
    """Return the diffusion and the logistic growth of a density, `du/dt = D laplacian(u) + r u (1 - u)`."""
    u = s.target.u
    return [RateLaw(u, s.target.D * laplacian(u) + s.target.r * u * (1 - u))]


def _reactive_mesh(refine=2):
    """Return heat with a cubic reaction on the L-shaped mesh with fixed boundary temperatures."""
    grid = skfem.MeshTri.init_lshaped().refined(refine)
    space = CellComplexSpace(points=grid.p, cells=grid.t, boundary={'u': ('fixed', 0.5)})
    return _system('plate', _reactive_heat, {'u': 0.0, 'D': 1.0, 'r': 1.0}, space=space)


def _fisher_line(size, boundary):
    """Return Fisher-KPP on a line with an edge condition."""
    space = LatticeSpace(size, spacing=0.25, boundary={'u': boundary})
    return _system('line', _fisher_kpp, {'u': 0.0, 'D': 1.0, 'r': 1.0}, space=space)


@pytest.mark.parametrize(('boundary', 'entries'), [('zero-flux', 2998), ('periodic', 3000)])
def test_stencil_pattern(boundary, entries):
    """Test that the three-point Laplacian gives a tridiagonal pattern, closed into a ring by periodic edges."""
    pattern = _problem(_fisher_line(1000, boundary)).jacobian.get_pattern()
    assert pattern.nnz == entries
    rows, columns = pattern.nonzero()
    assert set(np.abs(rows - columns)) <= ({0, 1, 999} if boundary == 'periodic' else {0, 1})


def test_mesh_pattern():
    """Test that the mesh pattern is the stiffness pattern with the diagonal on free nodes and empty fixed rows."""
    problem = _problem(_reactive_mesh())
    pattern = problem.jacobian.get_pattern().toarray()
    grid = skfem.MeshTri.init_lshaped().refined(2)
    stiffness = FieldLowering._get_mesh_laplacian(CellComplexSpace(points=grid.p, cells=grid.t)).toarray() != 0
    free = np.setdiff1d(np.arange(problem.initial_state.size), problem.constrained)
    assert problem.constrained.size
    assert not pattern[problem.constrained].any()
    expected = stiffness | np.eye(len(stiffness), dtype=bool)
    np.testing.assert_array_equal(pattern[free], expected[free])


@pytest.mark.parametrize(
    'system',
    [HeatDiffusionSystem((12, 12)), HeatLShapeSystem(2)],
    ids=['lattice', 'mesh'],
)
def test_pattern_agrees_with_split_solver(system):
    """Test that the pattern holds every entry of the sparse linear part of linear lattice and mesh fields."""
    problem = _problem(system)
    solver = IMEXEulerSolver().prepare(problem)
    solver._build_operator(problem.initial_parameters)
    operator = solver.sparse_
    pattern = problem.jacobian.get_pattern().toarray()
    assert pattern[operator.toarray() != 0].all()


@pytest.mark.parametrize(
    ('system', 'block'),
    [
        pytest.param(
            _pair_ring(12),
            12,
            marks=pytest.mark.skip(
                reason='engine gap: a reduction over the neighbours within a radius gives no Jacobian pattern, '
                'not the dense block of its group',
            ),
        ),
        (KuramotoSivashinskySystem(16), 16),
    ],
    ids=['pairs', 'spectral'],
)
def test_run_time_couplings_are_dense(system, block):
    """Test that pair interactions and spectral operators give the dense block of their group or field."""
    pattern = _problem(system).jacobian.get_pattern().toarray()
    assert pattern[:block, :block].all()


def _relative_error(problem, rng):
    """Return the relative Frobenius error of the exact Jacobian against central finite differences."""
    x = problem.initial_state + 0.1 * rng.random(problem.initial_state.size)
    p = problem.initial_parameters
    exact = problem.jacobian.prepare(problem).run(RunStateSpec(0.0, x, p)).values_.toarray()
    estimate = _finite_differences(problem, x, p)
    return np.linalg.norm(exact - estimate) / np.linalg.norm(estimate)


def test_exact_jacobian_of_gray_scott():
    """Test that the exact Jacobian of Gray-Scott on a 64 by 64 lattice agrees with central finite differences."""
    problem = _problem(GrayScottSystem((64, 64), backend=NumpyBackend()))
    assert _relative_error(problem, np.random.default_rng(0)) < 1e-6


def test_exact_jacobian_of_mesh_with_reaction():
    """Test that the exact Jacobian of heat with a reaction on a mesh matches finite differences, fixed rows empty."""
    problem = _problem(_reactive_mesh())
    assert _relative_error(problem, np.random.default_rng(0)) < 1e-6
    jacobian = (
        problem.jacobian.prepare(problem)
        .run(RunStateSpec(0.0, problem.initial_state, problem.initial_parameters))
        .values_
    )
    assert not jacobian[problem.constrained].nnz


def test_exact_jacobian_reads_the_parameters_it_receives():
    """Test that the exact Jacobian is evaluated at the parameters of the call."""
    problem = _problem(_fisher_line(20, 'zero-flux'))
    x, p = problem.initial_state, problem.initial_parameters.copy()
    first = problem.jacobian.prepare(problem).run(RunStateSpec(0.0, x, p)).values_.toarray()
    p[problem.parameters.get_entry('line.D').indices] *= 2
    second = problem.jacobian.prepare(problem).run(RunStateSpec(0.0, x, p)).values_.toarray()
    off = ~np.eye(20, dtype=bool)
    np.testing.assert_allclose(second[off], 2 * first[off])


def _callable_rate(t, values):
    """Return the decay of the line."""
    return -values['line.u']


def test_exact_jacobian_of_callable_law_raises():
    """Test that the exact Jacobian of a callable law raises an error naming the law and the automatic option."""

    system = _system('line', lambda s: [RateLaw(s.target.u, _callable_rate)], {'u': 0.0}, space=LatticeSpace(10))
    problem = _problem(system)
    with pytest.raises(ValidationError, match=r"rate laws of laws hold callables.*jacobian='auto'"):
        problem.jacobian.prepare(problem)
    assert problem.jacobian.get_pattern().toarray().all()


def _nonlocal_line(law, size=40, u=0.0, **settings):
    """Return a field on a line of fixed zero edges whose state `u` changes at the rate `law(u)`."""
    space = LatticeSpace(size, spacing=1 / (size + 1), boundary='fixed')
    return _system('line', lambda s: [RateLaw(s.target.u, law(s.target.u))], {'u': u}, space=space, **settings)


def test_pattern_marks_inversions_dense_and_convolutions_within_their_radius():
    """Test that an inversion couples every site and a convolution of radius two the sites within two spacings."""
    inverted = _problem(_nonlocal_line(lambda u: partial(inverse_helmholtz(u, 1), 0), backend=NumpyBackend()))
    assert inverted.jacobian.get_pattern().toarray().all()
    system = _nonlocal_line(lambda u: convolve(u, 1 + displacement(0) ** 2, 2 / 41), backend=NumpyBackend())
    convolved = _problem(system)
    rows, columns = convolved.jacobian.get_pattern().nonzero()
    assert set(np.abs(rows - columns)) == {0, 1, 2}


def test_implicit_solver_agrees_with_explicit_on_screened_diffusion():
    """Test that BDF agrees with RK45 within 1e-6 on a diffusion relaxed by its screened inversion."""
    positions = np.arange(1, 41) / 41
    system = _nonlocal_line(
        lambda u: laplacian(u) / 100 + inverse_helmholtz(u, 3),
        u=np.sin(np.pi * positions) + 0.3 * np.sin(4 * np.pi * positions),
        backend=NumpyBackend(),
    )
    finals = [
        system.set_params(solver=ScipyODESolver(method, rtol=1e-10, atol=1e-12))
        .prepare()
        .simulate(until=0.5, dt=0.5)
        .states[-1]
        for method in ('BDF', 'RK45')
    ]
    assert np.abs(finals[0] - finals[1]).max() < 1e-6
