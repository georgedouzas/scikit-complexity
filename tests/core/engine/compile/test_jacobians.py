"""Test the exact Jacobian of drifts that read reductions, and its sparsity pattern."""

from typing import Annotated

import numpy as np
import pytest
import skfem
from sage.all__sagemath_symbolics import sin, tanh

from skcomplex.core.components.elements import Element, FieldElement, GroupElement
from skcomplex.core.components.interactions import PairCouplingInteraction
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
    partial,
    total,
    variance,
    where,
)
from skcomplex.core.components.spaces import ContinuousSpace, LatticeSpace, MeshSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.backends import NumpyBackend
from skcomplex.core.engine.compile._field_lowering import FieldLowering
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import IMEXEulerSolver, ScipyODESolver
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RunStateSpec, StateSpec, VisualSpec
from skcomplex.systems.biology.ecology import FisherKPPElement
from skcomplex.systems.chemistry.pattern_formation import GrayScottSystem
from skcomplex.systems.physics.common.oscillators import SitedOscillatorElement
from skcomplex.systems.physics.fluids import KuramotoSivashinskySystem
from skcomplex.systems.physics.heat import HeatDiffusionSystem, HeatLShapeSystem

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
        lambda s: [RateLaw(s.x, -s.x + tanh(total(s.x)) + mean(s.x**2))],
        lambda s: [
            RateLaw(s.x, s.x * (total(s.x, weights=PAYOFF) - total(s.x * total(s.x, weights=PAYOFF)))),
            RateLaw(
                s.y,
                maximum(s.x * s.y)
                + variance(s.y)
                - minimum(sin(s.x))
                + mean(s.y, where=s.x > 0) * count(where=s.y > 0),
            ),
        ],
        lambda s: [RateLaw(s.x, where(s.x > 0, total(s.y**2), -s.x)), RateLaw(s.y, -s.y)],
    ],
    ids=['tanh of total', 'every kind', 'condition'],
)
def test_jacobian_against_finite_differences(laws):
    """Test that the exact Jacobian equals central finite differences and that its pattern covers every entry."""

    class MemberElement(Element):
        """Member element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
            y: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.y = y

        def _get_laws(self, s):
            """Return the laws of the member element."""
            return laws(s)

    system = System(
        elements=[
            GroupElement(
                'group',
                element=MemberElement,
                size=20,
                values={'x': RNG.normal(size=20), 'y': RNG.normal(size=20)},
            ),
        ],
    )
    problem = PassCompiler().prepare(system).run().problem_
    x, p = problem.initial_state, problem.initial_parameters
    exact = problem.jacobian.prepare(problem).run(RunStateSpec(0.0, x, p)).values_.toarray()
    estimate = _finite_differences(problem, x, p)
    np.testing.assert_allclose(exact, estimate, atol=1e-6)
    pattern = problem.jacobian.pattern().toarray()
    assert pattern[np.abs(estimate) > 1e-9].all()


def _pair_ring(size):
    """Return phase oscillators on a periodic ring, coupled by the sine of their phase differences within two sites."""
    values = {'theta': RNG.uniform(0, 2 * np.pi, size), 'x': np.arange(size, dtype=float).reshape(-1, 1)}
    ring = GroupElement('ring', element=SitedOscillatorElement, size=size, values=values)
    coupling = PairCouplingInteraction(
        'coupling',
        'ring',
        'ring',
        cutoff=2.5,
        write='theta',
        function=lambda s: sin(s.second.theta - s.first.theta),
        period=float(size),
    )
    return System(elements=[ring], interactions=[coupling], space=ContinuousSpace(n_dim=1))


def test_jacobian_of_pair_sums_is_not_derived():
    """Test that the exact Jacobian of a pair interaction raises an error that names the interaction."""
    problem = PassCompiler().prepare(_pair_ring(10)).run().problem_
    with pytest.raises(ValidationError, match=r'coupling.*finite differences'):
        problem.jacobian.prepare(problem).run(RunStateSpec(0.0, problem.initial_state, problem.initial_parameters))


def test_pattern_is_omitted_above_the_limit():
    """Test that the pattern of a dense block above the size limit is omitted."""

    class WideElement(Element):
        """Wide element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the wide element."""
            return [RateLaw(s.x, total(s.x))]

    MemberElement = WideElement
    system = System(elements=[GroupElement('group', element=MemberElement, size=3000)])
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.jacobian.pattern() is None


class ReactiveHeatElement(Element):
    """Reactive heat element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 0.0,
        D: Annotated[object, ParameterSpec()] = 1.0,
        r: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.D = D
        self.r = r

    def _get_laws(self, s):
        """Return the laws of the reactive heat element."""
        return [RateLaw(s.u, s.D * laplacian(s.u) + s.r * s.u * (1 - s.u**2))]


def _reactive_mesh(refine=2):
    """Return heat with a cubic reaction on the L-shaped mesh with fixed boundary temperatures."""
    grid = skfem.MeshTri.init_lshaped().refined(refine)
    plate = FieldElement(
        'plate',
        element=ReactiveHeatElement,
        space=MeshSpace(grid.p, grid.t),
        boundary={'u': ('fixed', 0.5)},
        discretization='finite-element',
    )
    return System(elements=[plate])


def _fisher_line(size, boundary):
    """Return Fisher-KPP on a line with an edge condition."""
    space = LatticeSpace(size, spacing=0.25)
    return System(elements=[FieldElement('line', element=FisherKPPElement, space=space, boundary={'u': boundary})])


@pytest.mark.parametrize(('boundary', 'entries'), [('zero-flux', 2998), ('periodic', 3000)])
def test_stencil_pattern(boundary, entries):
    """Test that the three-point Laplacian gives a tridiagonal pattern, closed into a ring by periodic edges."""
    pattern = PassCompiler().prepare(_fisher_line(1000, boundary)).run().problem_.jacobian.pattern()
    assert pattern.nnz == entries
    rows, columns = pattern.nonzero()
    assert set(np.abs(rows - columns)) <= ({0, 1, 999} if boundary == 'periodic' else {0, 1})


def test_mesh_pattern():
    """Test that the mesh pattern is the stiffness pattern with the diagonal on free nodes and empty fixed rows."""
    problem = PassCompiler().prepare(_reactive_mesh()).run().problem_
    pattern = problem.jacobian.pattern().toarray()
    grid = skfem.MeshTri.init_lshaped().refined(2)
    stiffness = FieldLowering._mesh_laplacian(MeshSpace(grid.p, grid.t)).toarray() != 0
    free = np.setdiff1d(np.arange(problem.initial_state.size), problem.constrained)
    assert problem.constrained.size
    assert not pattern[problem.constrained].any()
    expected = stiffness | np.eye(len(stiffness), dtype=bool)
    np.testing.assert_array_equal(pattern[free], expected[free])


@pytest.mark.parametrize(
    'system',
    [HeatDiffusionSystem((12, 12)).prepare().system_, HeatLShapeSystem(2).prepare().system_],
    ids=['lattice', 'mesh'],
)
def test_pattern_agrees_with_split_solver(system):
    """Test that the pattern holds every entry of the sparse linear part of linear lattice and mesh fields."""
    problem = PassCompiler().prepare(system).run().problem_
    solver = IMEXEulerSolver().prepare(problem)
    solver._build_operator(problem.initial_parameters)
    operator = solver.sparse_
    pattern = problem.jacobian.pattern().toarray()
    assert pattern[operator.toarray() != 0].all()


@pytest.mark.parametrize(
    ('system', 'block'),
    [(_pair_ring(12), 12), (KuramotoSivashinskySystem(16).prepare().system_, 16)],
    ids=['pairs', 'spectral'],
)
def test_run_time_couplings_are_dense(system, block):
    """Test that pair interactions and spectral operators give the dense block of their group or field."""
    pattern = PassCompiler().prepare(system).run().problem_.jacobian.pattern().toarray()
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
    problem = PassCompiler(backend=NumpyBackend()).prepare(GrayScottSystem((64, 64)).prepare().system_).run().problem_
    assert _relative_error(problem, np.random.default_rng(0)) < 1e-6


def test_exact_jacobian_of_mesh_with_reaction():
    """Test that the exact Jacobian of heat with a reaction on a mesh matches finite differences, fixed rows empty."""
    problem = PassCompiler().prepare(_reactive_mesh()).run().problem_
    assert _relative_error(problem, np.random.default_rng(0)) < 1e-6
    jacobian = (
        problem.jacobian.prepare(problem)
        .run(RunStateSpec(0.0, problem.initial_state, problem.initial_parameters))
        .values_
    )
    assert not jacobian[problem.constrained].nnz


def test_exact_jacobian_reads_the_parameters_it_receives():
    """Test that the exact Jacobian is evaluated at the parameters of the call."""
    problem = PassCompiler().prepare(_fisher_line(20, 'zero-flux')).run().problem_
    x, p = problem.initial_state, problem.initial_parameters.copy()
    first = problem.jacobian.prepare(problem).run(RunStateSpec(0.0, x, p)).values_.toarray()
    p[problem.parameters.entry('line.D').indices] *= 2
    second = problem.jacobian.prepare(problem).run(RunStateSpec(0.0, x, p)).values_.toarray()
    off = ~np.eye(20, dtype=bool)
    np.testing.assert_allclose(second[off], 2 * first[off])


def _callable_rate(t, values):
    """Return the decay of the line."""
    return -values['line.u']


def test_exact_jacobian_of_callable_law_raises():
    """Test that the exact Jacobian of a callable law raises an error naming the law and the automatic option."""

    class DecayElement(Element):
        """Decay element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            u: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.u = u

        def _get_laws(self, s):
            """Return the laws of the decay element."""
            return [RateLaw(s.u, _callable_rate)]

    system = System(elements=[FieldElement('line', element=DecayElement, space=LatticeSpace(10))])
    problem = PassCompiler().prepare(system).run().problem_
    with pytest.raises(ValidationError, match=r"line.*jacobian='auto'"):
        problem.jacobian.prepare(problem)
    assert problem.jacobian.pattern().toarray().all()


def _nonlocal_line(law, size=40, **settings):
    """Return a field on a line of fixed zero edges whose state changes at the rate `law(s)`."""

    class NonlocalElement(Element):
        """Nonlocal element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            u: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.u = u

        def _get_laws(self, s):
            """Return the law of the nonlocal element."""
            return [RateLaw(s.u, law(s))]

    space = LatticeSpace(size, spacing=1 / (size + 1), boundary='fixed')
    return System(elements=[FieldElement('line', element=NonlocalElement, space=space, **settings)])


def test_pattern_marks_inversions_dense_and_convolutions_within_their_radius():
    """Test that an inversion couples every site and a convolution of radius two the sites within two spacings."""
    system = _nonlocal_line(lambda s: partial(inverse_helmholtz(s.u, 1), 0))
    inverted = PassCompiler(backend=NumpyBackend()).prepare(system).run().problem_
    assert inverted.jacobian.pattern().toarray().all()
    system = _nonlocal_line(lambda s: convolve(s.u, 1 + displacement(0) ** 2, 2 / 41))
    convolved = PassCompiler(backend=NumpyBackend()).prepare(system).run().problem_
    rows, columns = convolved.jacobian.pattern().nonzero()
    assert set(np.abs(rows - columns)) == {0, 1, 2}


def test_implicit_solver_agrees_with_explicit_on_screened_diffusion():
    """Test that BDF agrees with RK45 within 1e-6 on a diffusion relaxed by its screened inversion."""
    positions = np.arange(1, 41) / 41
    system = _nonlocal_line(
        lambda s: laplacian(s.u) / 100 + inverse_helmholtz(s.u, 3),
        values={'u': np.sin(np.pi * positions) + 0.3 * np.sin(4 * np.pi * positions)},
    )
    finals = [
        simulate(
            system,
            until=0.5,
            dt=0.5,
            solver=ScipyODESolver(method, rtol=1e-10, atol=1e-12),
            backend=NumpyBackend(),
        ).states[-1]
        for method in ('BDF', 'RK45')
    ]
    assert np.abs(finals[0] - finals[1]).max() < 1e-6
