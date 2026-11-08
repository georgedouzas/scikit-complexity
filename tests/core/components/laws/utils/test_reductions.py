"""Test the `_reductions` module of `skcomplex.core.components.laws.utils`."""

import itertools
import math
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
import scipy.sparse
import scipy.special
from sage.all__sagemath_symbolics import RR, SR, atan2, cos, exp, pi, sin, sqrt, tanh, var, vector

from skcomplex.core.components.base import BasePart
from skcomplex.core.components.elements import Element, GroupElement, LatticeElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import EquationLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import (
    NEIGHBOR_REDUCTIONS,
    circular_mean,
    count,
    hyperedges,
    hyperedges_in,
    log_sum_exp,
    maximum,
    mean,
    minimum,
    neighborhoods_in,
    neighbors,
    reductions_in,
    total,
    uniform,
    variance,
)
from skcomplex.core.components.spaces import (
    ConeNeighborhood,
    ContinuousSpace,
    DiscreteTime,
    LatticeSpace,
    NearestNeighborhood,
    NetworkSpace,
    RadiusNeighborhood,
    StaticTime,
    VonNeumannNeighborhood,
)
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import JaxBackend, NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.problems._reduction_kernel import ReductionKernel
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ValidationError
from skcomplex.core.experiments import sweep
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


@pytest.mark.parametrize(
    ('function', 'kind'),
    [(total, 'sum'), (mean, 'mean'), (maximum, 'max'), (minimum, 'min'), (variance, 'var')],
    ids=['total', 'mean', 'maximum', 'minimum', 'variance'],
)
def test_reductions_stay_unevaluated(function, kind):
    """Test that a reduction of a member symbol stays an unevaluated call of its kind."""
    x = var('x')
    found = reductions_in(1 + function(x**2))
    assert [item.kind for item in found] == [kind]
    assert str(found[0].argument) == 'x^2'


def test_reductions_survive_simplify():
    """Test that reductions with weights and without conditions come back unchanged from Maxima."""
    x = var('x')
    weights = np.arange(9.0).reshape(3, 3)
    expression = -x + tanh(total(x)) + 2 * mean(cos(x)) + total(x, weights=weights) + count()
    for simplified in (expression.simplify(), expression.simplify_full()):
        assert {str(item.call) for item in reductions_in(simplified)} == {
            str(item.call) for item in reductions_in(expression)
        }


def test_reductions_record_condition_weights_and_group():
    """Test that a reduction holds its condition, its weights, and the group it names."""
    x, y = var('x'), var('y')
    weights = scipy.sparse.csr_matrix(np.eye(2))
    (found,) = reductions_in(mean(x, over=SimpleNamespace(t=var('t'), y=y), where=x > 0, weights=weights))
    assert str(found.condition) == 'x > 0'
    assert str(found.over) == str(y)
    assert (found.weights != weights).nnz == 0
    assert reductions_in(found.call)
    assert not reductions_in(x + 1)


def test_count_takes_no_expression():
    """Test that a count reads only its condition."""
    x = var('x')
    (found,) = reductions_in(count(where=x > 0.5))
    assert found.kind == 'count'
    assert found.argument == 1


@pytest.mark.parametrize(
    ('call', 'match'),
    [
        (lambda x: maximum(x, over='group'), 'namespace'),
        (lambda x: total(x, weights=np.ones((3, 3, 3))), 'matrix'),
    ],
    ids=['group name', 'tensor weights'],
)
def test_malformed_reductions_are_rejected(call, match):
    """Test that a group given by name and weights that are neither a matrix nor a shared row raise errors."""
    with pytest.raises(ValidationError, match=match):
        call(var('x'))


def test_count_reduces_values():
    """Test that the reduction table counts members, also none."""
    assert ReductionKernel._reduce_values([1.0, 2.0], 'count') == 2.0
    assert ReductionKernel._reduce_values([], 'count') == 0.0


BACKENDS = (NumpyBackend(), NumbaBackend())
GAMMA = 0.5
CRITICAL = 2 * GAMMA


class ReducingElement(Element):
    """Element whose state `out` is the reduction of the neighbour's `y` over a neighbourhood."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        out: Annotated[object, StateSpec()] = 0.0,
        y: Annotated[object, StateSpec()] = 0.0,
        over: object = None,
        reduce: str = 'sum',
    ) -> None:
        super().__init__(label, visual=visual)
        self.out = out
        self.y = y
        self.over = over
        self.reduce = reduce

    def _get_laws(self, s):
        """Return the update of `out` to the reduction."""
        return [UpdateLaw(s.out, neighbors(s.other.y, over=self.over, reduce=self.reduce))]


class PositionedReducingElement(ReducingElement):
    """Reducing element with a position in a continuous space."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        out: Annotated[object, StateSpec()] = 0.0,
        y: Annotated[object, StateSpec()] = 0.0,
        over: object = None,
        reduce: str = 'sum',
        x: Annotated[object, StateSpec(vector=True, role='position')] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, out=out, y=y, over=over, reduce=reduce)
        self.x = x


class HeadedReducingElement(PositionedReducingElement):
    """Reducing element with a position and a heading angle."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        out: Annotated[object, StateSpec()] = 0.0,
        y: Annotated[object, StateSpec()] = 0.0,
        over: object = None,
        reduce: str = 'sum',
        x: Annotated[object, StateSpec(vector=True, role='position')] = 0.0,
        phi: Annotated[object, StateSpec(domain=RR)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, out=out, y=y, over=over, reduce=reduce, x=x)
        self.phi = phi


def _expected(kind, values, weights):
    """Return the reduction of the values of one member's neighbours with their weights."""
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if not values.size:
        return 0.0
    products = weights * values
    reductions = {
        'sum': products.sum(),
        'mean': products.sum() / values.size,
        'count': (weights * (values != 0)).sum(),
        'max': products.max(),
        'min': products.min(),
        'normalized-mean': products.sum() / weights.sum() if weights.sum() else 0.0,
        'argmax': values[np.argmax(weights)],
    }
    return reductions[kind]


def _outputs(system, backend):
    """Return the values of `out` after one update of the first group of a system."""
    problem = PassCompiler(backend=backend).prepare(system).run().problem_
    after = problem.update(0.0, problem.initial_state, problem.initial_parameters)
    return problem.states.read(after, f'{system.elements[0].label}.out')


def _minimum_image(difference, side):
    """Return displacements wrapped to the minimum image in a periodic box of one side."""
    return difference - side * np.rint(difference / side)


def test_neighbors_keeps_the_values_at_lattice_offsets():
    """Test that a lattice neighbourhood without a reduction gives the values at every offset, for local energies."""
    values = neighbors(var('spin'), VonNeumannNeighborhood(), reduce='each')
    assert [str(item) for item in values] == [f'neighbor(spin, {index})' for index in range(4)]


def test_neighbors_rejects_an_unknown_reduction():
    """Test that an unknown reduction raises an error that lists the reductions."""
    with pytest.raises(ValidationError, match='normalized-mean'):
        neighbors(1, reduce='median')


def test_neighborhoods_in_finds_the_neighbourhood_and_the_weight():
    """Test that a reduction found in an expression keeps its neighbourhood, kind, and weight."""
    over = ConeNeighborhood(2.0, math.pi / 3, direction='phi')
    found = neighborhoods_in(1 + neighbors(2, over=over, reduce='normalized-mean', weight=3))
    assert [(item.over, item.kind, item.weight) for item in found] == [(over, 'normalized-mean', 3)]


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
@pytest.mark.parametrize('kind', NEIGHBOR_REDUCTIONS)
def test_reduction_over_a_network_matches_its_definition(kind, backend):
    """Test that every reduction over the neighbours of a network node matches the reduction of their values."""
    graph = nx.gnp_random_graph(12, 0.3, seed=1)
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    values = np.random.default_rng(0).choice([-2.0, 0.0, 1.0, 3.5], 12)
    system = System(
        elements=[GroupElement('g', element=ReducingElement, size=12, values={'y': values, 'reduce': kind})],
        space=NetworkSpace(graph),
    )
    expected = [_expected(kind, values[list(graph.neighbors(node))], np.ones(graph.degree(node))) for node in graph]
    assert np.allclose(_outputs(system, backend), expected, atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_weighted_reduction_reads_the_member_and_its_neighbour(backend):
    """Test that a weight of the member and the neighbour weighs every neighbour in a normalized mean."""

    class WeightedElement(Element):
        """Weighted element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            out: Annotated[object, StateSpec()] = 0.0,
            y: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.out = out
            self.y = y

        def _get_laws(self, s):
            """Return the laws of the weighted element."""
            return [UpdateLaw(s.out, neighbors(s.other.y, reduce='normalized-mean', weight=s.y + s.other.y))]

    weighted = WeightedElement
    values = np.array([1.0, 2.0, 3.0, 4.0])
    system = System(
        elements=[GroupElement('g', element=weighted, size=4, values={'y': values})],
        space=NetworkSpace(nx.path_graph(4)),
    )
    expected = [
        _expected('normalized-mean', values[others], values[member] + values[others])
        for member, others in enumerate([[1], [0, 2], [1, 3], [2]])
    ]
    assert np.allclose(_outputs(system, backend), expected, atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_reduction_over_a_lattice_leaves_out_sites_outside_fixed_edges(backend):
    """Test that a sum over a von Neumann neighbourhood of a fixed lattice sums the neighbours inside the lattice."""
    values = np.arange(12.0).reshape(3, 4)
    space = LatticeSpace((3, 4), boundary='fixed')
    given = {'y': values, 'over': VonNeumannNeighborhood()}
    system = System(elements=[LatticeElement('g', element=ReducingElement, space=space, values=given)])
    padded = np.pad(values, 1)
    expected = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    assert np.allclose(_outputs(system, backend), expected.ravel(), atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_reduction_without_a_neighbourhood_reads_the_lattice_neighbourhood(backend):
    """Test that a reduction without a neighbourhood on a periodic lattice reads the lattice's Moore neighbourhood."""
    values = np.random.default_rng(2).normal(size=(4, 5))
    system = System(
        elements=[
            LatticeElement('g', element=ReducingElement, space=LatticeSpace((4, 5)), values={'y': values}),
        ],
    )
    expected = sum(np.roll(values, (a - 1, b - 1), axis=(0, 1)) for a, b in np.ndindex(3, 3)) - values
    assert np.allclose(_outputs(system, backend), expected.ravel(), atol=1e-12)


def _scattered(count, side, seed):
    """Return positions, values, and heading angles of members scattered in a square."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, side, (count, 2)), rng.normal(size=count), rng.uniform(-math.pi, math.pi, count)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
@pytest.mark.parametrize('boundary', ['periodic', 'open'])
@pytest.mark.parametrize('center', [False, True])
def test_reduction_within_a_radius_matches_a_direct_search(boundary, center, backend):
    """Test that a sum over the members within a radius matches a search of every pair, with minimum images."""
    side, radius = 4.0, 0.9
    positions, values, _ = _scattered(80, side, 3)
    given = {'x': positions, 'y': values, 'over': RadiusNeighborhood(radius, center=center)}
    space = ContinuousSpace(extent=(side, side), boundary=boundary)
    system = System(
        elements=[GroupElement('g', element=PositionedReducingElement, size=80, values=given)],
        space=space,
    )
    difference = positions[:, None, :] - positions[None, :, :]
    if boundary == 'periodic':
        difference = _minimum_image(difference, side)
    within = np.linalg.norm(difference, axis=2) <= radius
    np.fill_diagonal(within, center)
    assert np.allclose(_outputs(system, backend), within @ values, atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
@pytest.mark.parametrize('boundary', ['periodic', 'open'])
def test_reduction_over_the_nearest_members_matches_a_direct_search(boundary, backend):
    """Test that a mean over the k nearest members matches a sort of every distance."""
    side, k = 5.0, 6
    positions, values, _ = _scattered(70, side, 4)
    given = {'x': positions, 'y': values, 'over': NearestNeighborhood(k), 'reduce': 'mean'}
    space = ContinuousSpace(extent=(side, side), boundary=boundary)
    system = System(
        elements=[GroupElement('g', element=PositionedReducingElement, size=70, values=given)],
        space=space,
    )
    difference = positions[:, None, :] - positions[None, :, :]
    if boundary == 'periodic':
        difference = _minimum_image(difference, side)
    distances = np.linalg.norm(difference, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1)[:, :k]
    assert np.allclose(_outputs(system, backend), values[nearest].mean(axis=1), atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_reduction_inside_a_cone_keeps_the_members_ahead(backend):
    """Test that a count over a cone of a heading angle counts the members within the radius and the half-angle."""
    side, radius, angle = 4.0, 1.2, math.pi / 3
    positions, values, headings = _scattered(90, side, 5)
    over = ConeNeighborhood(radius, angle, direction='phi')
    given = {'x': positions, 'y': values, 'phi': headings, 'over': over, 'reduce': 'count'}
    space = ContinuousSpace(extent=(side, side), boundary='periodic')
    system = System(
        elements=[GroupElement('g', element=HeadedReducingElement, size=90, values=given)],
        space=space,
    )
    direction = -_minimum_image(positions[:, None, :] - positions[None, :, :], side)
    distances = np.linalg.norm(direction, axis=2)
    heading = np.stack([np.cos(headings), np.sin(headings)], axis=1)
    cosines = np.einsum('ik,ijk->ij', heading, direction) / np.where(distances > 0, distances, 1)
    inside = (distances <= radius) & (cosines >= math.cos(angle))
    np.fill_diagonal(inside, False)
    assert np.allclose(_outputs(system, backend), inside.sum(axis=1), atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_neighbour_positions_are_read_by_minimum_image(backend):
    """Test that the position of a neighbour across a periodic edge is its image nearest to the member."""

    class ReadingElement(Element):
        """Reading element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            out: Annotated[object, StateSpec()] = 0.0,
            x: Annotated[object, StateSpec(vector=True, role='position')] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.out = out
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the reading element."""
            return [UpdateLaw(s.out, neighbors(s.other.x[0] - s.x[0], over=RadiusNeighborhood(1.0)))]

    reading = ReadingElement
    positions = np.array([[0.2, 0.5], [9.9, 0.5]])
    space = ContinuousSpace(extent=(10.0, 10.0), boundary='periodic')
    system = System(elements=[GroupElement('g', element=reading, size=2, values={'x': positions})], space=space)
    assert np.allclose(_outputs(system, backend), [-0.3, 0.3], atol=1e-12)


def test_neighbourhood_reads_a_place_by_its_role_and_not_a_state_named_x():
    """Test that a vector state named `x` without a role gives a neighbourhood no position to measure from."""

    class UnplacedElement(ReducingElement):
        """Reducing element with a vector state `x` that has no role."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            out: Annotated[object, StateSpec()] = 0.0,
            y: Annotated[object, StateSpec()] = 0.0,
            over: object = None,
            reduce: str = 'sum',
            x: Annotated[object, StateSpec(vector=True)] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual, out=out, y=y, over=over, reduce=reduce)
            self.x = x

    group = GroupElement('g', element=UnplacedElement, size=2, values={'over': RadiusNeighborhood(1.0)})
    with pytest.raises(ValidationError, match='position'):
        PassCompiler().prepare(System(elements=[group], space=ContinuousSpace(extent=(10.0, 10.0)))).run()


def test_interaction_reduces_over_the_neighbours_of_its_group():
    """Test that an interaction reads the neighbours of the group bound to its role through `s.target.other`."""

    class DiffusionInteraction(Interaction):
        """Diffusion interaction."""

        def __init__(
            self,
            label: str,
            *,
            target: Annotated[object, RoleSpec()],
        ) -> None:
            super().__init__(label)
            self.target = target

        def _get_laws(self, s):
            """Return the laws of the diffusion interaction."""
            return [RateLaw(s.target.y, neighbors(s.target.other.y - s.target.y))]

    diffusion = DiffusionInteraction

    class PlainElement(Element):
        """Plain element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            y: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.y = y

    plain = PlainElement
    system = System(
        elements=[GroupElement('g', element=plain, size=3, values={'y': [0.0, 1.0, 4.0]})],
        interactions=[diffusion('d', target='g')],
        space=NetworkSpace(nx.path_graph(3)),
    )
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [1.0, 2.0, -3.0]


def test_neighbour_symbols_outside_a_reduction_are_rejected():
    """Test that a law reading `s.other` outside a neighbourhood reduction raises an error that names `neighbors`."""

    class StrayElement(Element):
        """Stray element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            y: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.y = y

        def _get_laws(self, s):
            """Return the laws of the stray element."""
            return [RateLaw(s.y, s.other.y)]

    stray = StrayElement
    system = System(elements=[GroupElement('g', element=stray, size=3)], space=NetworkSpace(nx.path_graph(3)))
    with pytest.raises(ValidationError, match='neighbors'):
        PassCompiler().prepare(system).run()


def test_validate_reports_a_metric_neighbourhood_on_a_network():
    """Test that a law reducing over a radius in a system on a network space has a law-space issue."""
    system = System(
        elements=[GroupElement('g', element=ReducingElement, size=5, values={'over': RadiusNeighborhood(1.0)})],
        space=NetworkSpace(nx.cycle_graph(5)),
        time=DiscreteTime(),
    )
    issues = [issue for issue in PassCompiler().prepare(system).report_.issues if issue.code == 'law-space']
    assert len(issues) == 1
    assert 'continuous space' in issues[0].fix
    assert 'over=None' in issues[0].fix


def test_validate_reports_a_network_neighbourhood_in_a_continuous_space():
    """Test that a reduction over the neighbours of the space in a continuous space has a law-space issue."""
    system = System(
        elements=[GroupElement('g', element=PositionedReducingElement, size=5)],
        space=ContinuousSpace(extent=(2.0, 2.0)),
        time=DiscreteTime(),
    )
    assert [issue.code for issue in PassCompiler().prepare(system).report_.issues] == ['law-space']


def test_validate_reports_a_cone_without_a_heading():
    """Test that a cone over members without a velocity or heading state has a law-domain issue."""
    over = ConeNeighborhood(1.0, 1.0)
    system = System(
        elements=[GroupElement('g', element=PositionedReducingElement, size=5, values={'over': over})],
        space=ContinuousSpace(extent=(2.0, 2.0)),
        time=DiscreteTime(),
    )
    issues = PassCompiler().prepare(system).report_.issues
    assert [issue.code for issue in issues] == ['law-domain']
    assert 'order=2' in issues[0].fix


def test_validate_accepts_a_cone_along_the_velocity():
    """Test that a cone reads the first derivative of a position of order 2 as the heading of every member."""

    class MovingElement(Element):
        """Moving element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            out: Annotated[object, StateSpec()] = 0.0,
            y: Annotated[object, StateSpec()] = 0.0,
            x: Annotated[object, StateSpec(vector=True, order=2, role='position')] = 0.0,
            x_d1: object = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.out = out
            self.y = y
            self.x = x
            self.x_d1 = x_d1

        def _get_laws(self, s):
            """Return the laws of the moving element."""
            return [RateLaw(s.out, neighbors(s.other.y, over=ConeNeighborhood(1.0, 1.0)))]

    moving = MovingElement
    system = System(elements=[GroupElement('g', element=moving, size=5)], space=ContinuousSpace(extent=(2.0, 2.0)))
    assert PassCompiler().prepare(system).report_.ok


class OscillatorElement(Element):
    """Oscillator element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec()] = 0.0,
        omega: Annotated[object, ParameterSpec()] = 0.0,
        K: Annotated[object, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.omega = omega
        self.K = K

    def _get_laws(self, s):
        """Return the laws of the oscillator element."""
        return [RateLaw(s.theta, s.omega + s.K * neighbors(sin(s.other.theta - s.theta), reduce='mean'))]


def _kuramoto(coupling, size=1000):
    """Return oscillators with Lorentzian frequencies of half-width `GAMMA` on a complete graph."""
    quantiles = (np.arange(size) + 0.5) / size
    omega = GAMMA * np.tan(math.pi * (quantiles - 0.5))
    theta = np.random.default_rng(1).uniform(0, 2 * math.pi, size)
    oscillators = GroupElement(
        'oscillators',
        element=OscillatorElement,
        size=size,
        values={'theta': theta, 'omega': omega, 'K': coupling},
    )
    return System(elements=[oscillators], space=NetworkSpace(nx.complete_graph(size)))


def _order(phases):
    """Return the modulus of the mean phase vector of every frame."""
    return np.abs(np.exp(1j * phases).mean(axis=1))


@pytest.mark.slow
def test_kuramoto_synchronizes_above_the_critical_coupling():
    """Test that 1000 oscillators on a complete graph stay incoherent below `2 gamma` and synchronize above it."""
    below = simulate(_kuramoto(0.5 * CRITICAL), until=10.0, dt=0.5, solver=RK4Solver(dt=0.1), backend=NumbaBackend())
    above = simulate(_kuramoto(3 * CRITICAL), until=10.0, dt=0.5, solver=RK4Solver(dt=0.1), backend=NumbaBackend())
    weak, strong = _order(below.values('oscillators.theta')), _order(
        above.values('oscillators.theta'),
    )
    assert weak[-5:].max() < 0.1
    assert strong[-5:].min() > 0.7
    assert strong[-1] > 10 * strong[0]


def test_kuramoto_agrees_across_backends():
    """Test that the Kuramoto run on a complete graph gives the same phases on NumPy and Numba."""
    runs = [
        simulate(_kuramoto(3 * CRITICAL), until=2.0, dt=0.5, solver=RK4Solver(dt=0.1), backend=backend)
        for backend in BACKENDS
    ]
    assert np.allclose(runs[0].values('oscillators.theta'), runs[1].values('oscillators.theta'), rtol=0, atol=1e-12)


RADIUS = 1.0


class MoverElement(Element):
    """Mover element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(vector=True, role='position')] = 0.0,
        theta: Annotated[object, StateSpec(period=2 * pi)] = 0.0,
        v0: Annotated[object, ParameterSpec()] = 0.03,
        eta: Annotated[object, ParameterSpec()] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.theta = theta
        self.v0 = v0
        self.eta = eta

    def _get_laws(self, s):
        """Return the laws of the mover element."""
        return [
            UpdateLaw(
                s.theta,
                atan2(
                    neighbors(sin(s.other.theta), over=RadiusNeighborhood(RADIUS, center=True)),
                    neighbors(cos(s.other.theta), over=RadiusNeighborhood(RADIUS, center=True)),
                )
                + uniform('noise', -s.eta / 2, s.eta / 2),
            ),
            UpdateLaw(s.x, s.x + s.v0 * vector([cos(s.theta), sin(s.theta)])),
        ]


def _vicsek(size=400, side=5.0):
    """Return the Vicsek system at high density in a periodic box under discrete time."""
    rng = np.random.default_rng(3)
    movers = GroupElement(
        'movers',
        element=MoverElement,
        size=size,
        values={'x': rng.uniform(0, side, (size, 2)), 'theta': rng.uniform(-math.pi, math.pi, size)},
    )
    return System(
        elements=[movers],
        space=ContinuousSpace(extent=(side, side), boundary='periodic'),
        time=DiscreteTime(),
    )


def test_vicsek_flocks_at_low_noise_and_high_density():
    """Test that the polar order parameter of the Vicsek system exceeds 0.9 at low noise and high density."""
    trajectory = simulate(_vicsek(), until=100, dt=1.0, random_state=0, backend=NumbaBackend())
    order = _order(trajectory.values('movers.theta'))
    assert order[0] < 0.2
    assert order[-1] > 0.9


def test_vicsek_agrees_across_backends():
    """Test that the Vicsek system gives the same headings and positions on NumPy and Numba for one seed."""
    runs = [simulate(_vicsek(), until=50, dt=1.0, random_state=0, backend=backend) for backend in BACKENDS]
    turns = np.angle(np.exp(1j * (runs[0].values('movers.theta') - runs[1].values('movers.theta'))))
    assert np.abs(turns).max() < 1e-12
    assert np.allclose(runs[0].values('movers.x'), runs[1].values('movers.x'), rtol=0, atol=1e-12)


class PhaseElement(Element):
    """Element with a phase."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class PairwiseInteraction(Interaction):
    """Rate of every member given by a reduction over its group that reads the member and the other member."""

    def __init__(
        self,
        label: str,
        *,
        group: Annotated[object, RoleSpec()],
        reduction: object = None,
    ) -> None:
        super().__init__(label)
        self.group = group
        self.reduction = reduction

    def _get_laws(self, s):
        """Return the rate of every member."""
        return [RateLaw(s.group.x, self.reduction(s.group))]


PAIRWISE = {
    'mean': lambda g: mean(sin(g.other.x - g.x), over=g),
    'total': lambda g: total(g.other.x * g.x, over=g),
    'selected-mean': lambda g: mean(g.other.x - g.x, over=g, where=g.other.x > g.x),
    'count': lambda g: count(over=g, where=g.other.x > g.x),
    'maximum': lambda g: maximum(cos(g.other.x - g.x), over=g),
    'minimum': lambda g: minimum(g.other.x - g.x, over=g),
}


def _pairwise_expected(name, values):
    """Return the reduction of every member over the differences to every member of a group."""
    differences = values[None, :] - values[:, None]
    selected = differences > 0
    sums = (differences * selected).sum(axis=1)
    expected = {
        'mean': np.sin(differences).mean(axis=1),
        'total': values * values.sum(),
        'selected-mean': np.divide(sums, selected.sum(axis=1), out=np.zeros(values.size), where=selected.any(axis=1)),
        'count': selected.sum(axis=1),
        'maximum': np.cos(differences).max(axis=1),
        'minimum': differences.min(axis=1),
    }
    return expected[name]


def _pairwise_drift(values, reduction, backend):
    """Return the rate of every member of a group whose rate is a reduction over the group."""
    system = System(
        elements=[GroupElement('g', element=PhaseElement, size=len(values), values={'x': values})],
        interactions=[PairwiseInteraction('pairs', group='g', reduction=reduction)],
    )
    problem = PassCompiler(backend=backend).prepare(system).run().problem_
    return problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'g.x')


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_mean_of_a_function_of_the_other_and_the_reading_member_matches_numpy(backend):
    """Test that the mean of `sin(other.x - x)` over a group of three gives `[0.584, 0, -0.584]` for phases 0, 1, 2."""
    values = np.array([0.0, 1.0, 2.0])
    drift = _pairwise_drift(values, PAIRWISE['mean'], backend)
    assert np.allclose(drift, np.sin(values[None, :] - values[:, None]).mean(axis=1), atol=1e-12)
    assert np.allclose(drift, [0.584, 0.0, -0.584], atol=1e-3)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
@pytest.mark.parametrize('name', list(PAIRWISE))
def test_reductions_of_the_other_member_read_every_pair(name, backend):
    """Test that every reduction that reads the other member and the reading member reduces over every pair."""
    values = np.random.default_rng(3).normal(size=7)
    drift = _pairwise_drift(values, PAIRWISE[name], backend)
    assert np.allclose(drift, _pairwise_expected(name, values), atol=1e-12)


def test_reduction_of_the_other_member_alone_is_the_group_reduction():
    """Test that a mean of the other member alone equals the mean of the group, the same for every member."""
    drift = _pairwise_drift([0.0, 1.0, 2.0, 5.0], lambda g: mean(g.other.x, over=g), NumpyBackend())
    assert drift.tolist() == [2.0] * 4


def test_reduction_of_the_other_member_rejects_constant_weights_of_pairs():
    """Test that a term of both members that does not factor, with a constant weight matrix, raises an error."""
    with pytest.raises(ValidationError, match='shaped parameter'):
        _pairwise_drift([0.0, 1.0], lambda g: total(sin(g.other.x - g.x), weights=np.eye(2)), NumpyBackend())


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_reduction_of_the_other_member_weights_terms_that_factor(backend):
    """Test that a weighted term of both members that factors gives `sum_j W_ij (x_j - x_i)`."""
    x, weights = np.array([0.0, 1.0, 3.0]), np.arange(9.0).reshape(3, 3)
    drift = _pairwise_drift(x, lambda g: total(g.other.x - g.x, weights=weights), backend)
    np.testing.assert_allclose(drift, weights @ x - x * weights.sum(axis=1), atol=1e-12)


class ImitatingElement(Element):
    """Element whose `out` is the strategy of the neighbour with the largest payoff."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        out: Annotated[object, StateSpec()] = 0.0,
        strategy: Annotated[object, StateSpec()] = 0.0,
        payoff: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.out = out
        self.strategy = strategy
        self.payoff = payoff

    def _get_laws(self, s):
        """Return the update of `out` to the strategy of the best neighbour."""
        return [UpdateLaw(s.out, neighbors(s.other.strategy, reduce='argmax', weight=s.other.payoff))]


NETWORK = nx.gnp_random_graph(12, 0.3, seed=5)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
@pytest.mark.parametrize(
    ('space', 'graph'),
    [(LatticeSpace(12), nx.cycle_graph(12)), (NetworkSpace(NETWORK), NETWORK)],
    ids=['ring', 'network'],
)
def test_argmax_returns_the_strategy_of_the_neighbour_with_the_largest_payoff(space, graph, backend):
    """Test that the argmax reduction gives the strategy of the best paid neighbour on a ring and a network."""
    rng = np.random.default_rng(4)
    strategies, payoffs = rng.integers(0, 5, 12).astype(float), rng.normal(size=12)
    values = {'strategy': strategies, 'payoff': payoffs}
    system = System(
        elements=[LatticeElement('g', element=ImitatingElement, space=space, values=values)],
        time=DiscreteTime(),
    )
    expected = [
        strategies[max(graph.neighbors(node), key=lambda other: payoffs[other])] if graph.degree(node) else 0.0
        for node in range(12)
    ]
    assert np.array_equal(_outputs(system, backend), expected)


class KeyedElement(Element):
    """Element with a value and a key."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        k: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.k = k


def _keyed_drift(values, keys, reduction, backend):
    """Return the rate of every member of a group of keyed members whose rate is a reduction over the group."""
    system = System(
        elements=[GroupElement('g', element=KeyedElement, size=len(values), values={'x': values, 'k': keys})],
        interactions=[PairwiseInteraction('pairs', group='g', reduction=reduction)],
    )
    problem = PassCompiler(backend=backend).prepare(system).run().problem_
    return problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'g.x')


KEYED = {
    'total': (lambda g: total(g.other.x, over=g, by='k'), np.sum),
    'mean': (lambda g: mean(g.other.x, over=g, by='k'), np.mean),
    'count': (lambda g: count(over=g, by='k'), np.size),
    'circular_mean': (
        lambda g: circular_mean(g.other.x, over=g, by='k'),
        lambda shared: np.arctan2(np.sin(shared).sum(), np.cos(shared).sum()),
    ),
}


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
@pytest.mark.parametrize('name', list(KEYED))
def test_reductions_keyed_by_a_state_reduce_over_the_members_sharing_its_value(name, backend):
    """Test that every reduction keyed by a state reduces over the members whose key equals the reading member's."""
    rng = np.random.default_rng(5)
    values, keys = rng.normal(size=9), rng.choice([0.0, 1.0, 2.0], 9)
    reduction, function = KEYED[name]
    expected = [function(values[keys == key]) for key in keys]
    assert np.allclose(_keyed_drift(values, keys, reduction, backend), expected, atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_keyed_reduction_combines_its_key_with_a_condition(backend):
    """Test that a keyed sum with a condition sums the members that share the key and meet the condition."""
    values, keys = np.array([1.0, -2.0, 3.0, 4.0, -5.0]), np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    drift = _keyed_drift(values, keys, lambda g: total(g.other.x, over=g, where=g.other.x > 0, by='k'), backend)
    assert drift.tolist() == [4.0, 4.0, 4.0, 4.0, 4.0]


@pytest.mark.parametrize(
    'call',
    [lambda x: total(x, by='k'), lambda x: count(over=SimpleNamespace(k=x), by='k')],
    ids=['no group', 'no other member'],
)
def test_keyed_reduction_needs_the_namespace_of_its_group(call):
    """Test that a reduction keyed by a state without the namespace of a group and its other member raises an error."""
    with pytest.raises(ValidationError, match='keyed by a state'):
        call(var('x'))


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_circular_mean_is_the_direction_of_the_sum_of_unit_vectors(backend):
    """Test that the circular mean of angles near the cut at pi stays near pi, unlike their arithmetic mean."""
    values = np.array([3.0, -3.0, 3.1, -3.1, 2.9])
    drift = _pairwise_drift(values, lambda g: circular_mean(g.other.x, over=g), backend)
    expected = np.arctan2(np.sin(values).sum(), np.cos(values).sum())
    assert np.allclose(drift, expected, atol=1e-12)
    assert abs(expected) > 3.0


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_circular_mean_over_network_neighbours_matches_its_definition(backend):
    """Test that the circular mean over the neighbours of a network node is the direction of their unit vectors."""
    graph = nx.gnp_random_graph(12, 0.3, seed=1)
    values = np.random.default_rng(0).uniform(-math.pi, math.pi, 12)
    given = {'y': values, 'reduce': 'circular-mean'}
    system = System(
        elements=[GroupElement('g', element=ReducingElement, size=12, values=given)],
        space=NetworkSpace(graph),
    )
    neighbours = [values[list(graph.neighbors(node))] for node in graph]
    expected = [np.arctan2(np.sin(item).sum(), np.cos(item).sum()) for item in neighbours]
    assert np.allclose(_outputs(system, backend), expected, atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
@pytest.mark.parametrize(('function', 'expected'), [(minimum, np.minimum), (maximum, np.maximum)])
def test_minimum_and_maximum_of_several_expressions_compare_member_by_member(function, expected, backend):
    """Test that the minimum and maximum of several expressions match NumPy's member by member."""
    values = np.random.default_rng(6).normal(size=8)
    drift = _pairwise_drift(values, lambda g: function(g.x, 0.1 - g.x, 0.2), backend)
    assert np.allclose(drift, expected(expected(values, 0.1 - values), 0.2), atol=1e-12)


def test_minimum_of_several_expressions_takes_no_group():
    """Test that the minimum of several expressions with a condition raises an error."""
    x = var('x')
    with pytest.raises(ValidationError, match='member by member'):
        minimum(x, 2 * x, where=x > 0)


HYPEREDGES = [(0, 1, 2), (1, 2, 3, 4), (3, 4, 5), (0, 2, 4, 5), (2, 5)]


class AroundElement(Element):
    """Element whose rate of `x` is a reduction over the hyperedges around it."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        term: str = 'triad',
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.term = term

    def _get_laws(self, s):
        """Return the rate of `x`, a reduction over the hyperedges around the member."""
        terms = {
            'triad': lambda: hyperedges(sin(s.others[0].x + s.others[1].x - 2 * s.x), size=3),
            'pull': lambda: hyperedges(total(s.members.x, over=s.members) - count(over=s.members) * s.x),
            'mean': lambda: hyperedges(mean(s.members.x, over=s.members) - s.x, reduce='mean'),
            'largest': lambda: hyperedges(s.others[0].x, reduce='max', size=2),
            'beyond': lambda: hyperedges(s.others[1].x, size=2),
            'extremum': lambda: hyperedges(maximum(s.members.x, over=s.members), reduce='max'),
            'missing': lambda: hyperedges(s.x, size=6),
            'count': lambda: hyperedges(s.x, reduce='count'),
        }
        return [RateLaw(s.x, terms[self.term]())]


def _around_drift(term, values, backend):
    """Return the drift of the members of the network of the hyperedges under a reduction over hyperedges."""
    space = NetworkSpace(nx.complete_graph(6), hyperedges=HYPEREDGES)
    system = System(
        elements=[
            LatticeElement('nodes', element=AroundElement, space=space, values={'x': values, 'term': term}),
        ],
    )
    problem = PassCompiler(backend=backend).prepare(system).run().problem_
    return problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'nodes.x')


def _around_expected(term, values):
    """Return the reduction over the hyperedges around every member computed by a loop over the hyperedges."""
    expected, numbers = np.zeros(6), np.zeros(6)
    for hyperedge in HYPEREDGES:
        for member in hyperedge:
            others = [values[other] for other in hyperedge if other != member]
            if term == 'triad' and len(hyperedge) == 3:
                expected[member] += np.sin(sum(others) - 2 * values[member])
            elif term == 'pull':
                expected[member] += values[list(hyperedge)].sum() - len(hyperedge) * values[member]
            elif term == 'mean':
                expected[member] += values[list(hyperedge)].mean() - values[member]
                numbers[member] += 1
            elif term == 'largest' and len(hyperedge) == 2:
                expected[member] = max(expected[member], others[0]) if numbers[member] else others[0]
                numbers[member] += 1
    return expected / np.maximum(numbers, 1) if term == 'mean' else expected


def test_hyperedges_stays_unevaluated_with_its_size_and_weight():
    """Test that a reduction over hyperedges keeps its kind, size, and weight, and maps a vector by component."""
    x, w = var('x w')
    (found,) = hyperedges_in(1 + hyperedges(x**2, reduce='mean', size=3, weight=w))
    assert (found.kind, found.size, str(found.argument), str(found.weight)) == ('mean', 3, 'x^2', 'w')
    (every,) = hyperedges_in(hyperedges(x))
    assert (every.size, every.weight) == (None, None)
    assert len(hyperedges(vector([x, w]))) == 2
    assert 'arctan2' in str(hyperedges(x, reduce='circular-mean'))


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [({'reduce': 'median'}, 'Unknown hyperedge reduction'), ({'size': 1}, 'size of 2 or more')],
)
def test_hyperedges_rejects_an_unknown_reduction_or_size(arguments, match):
    """Test that an unknown reduction or a size below two raises an error."""
    with pytest.raises(ValidationError, match=match):
        hyperedges(var('x'), **arguments)


@pytest.mark.parametrize('term', ['triad', 'pull', 'mean', 'largest'])
@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_hyperedges_matches_a_loop_over_the_hyperedges_around_every_member(term, backend):
    """Test that a reduction over the hyperedges around every member equals a loop over the hyperedges."""
    values = np.random.default_rng(4).uniform(0.0, 2 * np.pi, size=6)
    drift = _around_drift(term, values, backend)
    assert np.allclose(drift, _around_expected(term, values), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    ('term', 'match'),
    [('beyond', 'other members'), ('extremum', 'only the sum, the mean, and the count')],
)
def test_hyperedges_rejects_reads_its_hyperedges_do_not_have(term, match):
    """Test that reading more other members than a size has, or the maximum over every size, raises an error."""
    with pytest.raises(ValidationError, match=match):
        _around_drift(term, np.zeros(6), NumpyBackend())


def test_validate_reports_a_hyperedge_reduction_of_a_missing_size():
    """Test that a reduction over hyperedges of a size the network does not have gives a binding issue."""
    space = NetworkSpace(nx.complete_graph(6), hyperedges=HYPEREDGES)
    system = System(elements=[LatticeElement('nodes', element=AroundElement, space=space, values={'term': 'missing'})])
    (issue,) = [issue for issue in PassCompiler().prepare(system).report_.issues if issue.code == 'binding']
    assert 'size 6' in issue.message


def test_validate_reports_a_hyperedge_reduction_in_a_lattice():
    """Test that a reduction over hyperedges in a lattice space has a law-space issue."""
    space = LatticeSpace((3, 3))
    system = System(elements=[LatticeElement('sites', element=AroundElement, space=space, values={'term': 'count'})])
    issues = PassCompiler().prepare(system).report_.issues
    assert [issue.code for issue in issues] == ['law-space']
    assert 'hyperedges' in issues[0].message


def test_hyperedges_over_a_neighbourhood_keeps_it():
    """Test that a reduction over neighbour hyperedges records its neighbourhood and its size."""
    (found,) = hyperedges_in(hyperedges(var('x'), size=3, over=RadiusNeighborhood(1.5)))
    assert (found.size, repr(found.over)) == (3, 'RadiusNeighborhood(radius=1.5)')
    (network,) = hyperedges_in(hyperedges(var('x'), size=3))
    assert network.over is None


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'over': RadiusNeighborhood(1.0)}, 'size of 2 or more'),
        ({'size': 3, 'over': VonNeumannNeighborhood()}, 'continuous space'),
    ],
)
def test_hyperedges_over_a_neighbourhood_needs_a_size_and_a_metric_neighbourhood(arguments, match):
    """Test that neighbour hyperedges without a size, or over a lattice neighbourhood, raise an error."""
    with pytest.raises(ValidationError, match=match):
        hyperedges(var('x'), **arguments)


class NeighbourSetElement(Element):
    """Element in a continuous space whose rate of `y` is a reduction over its neighbour hyperedges."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(vector=True, role='position')] = 0.0,
        y: Annotated[object, StateSpec()] = 0.0,
        term: str = 'flank',
        radius: float = 1.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.term = term
        self.radius = radius

    def _get_laws(self, s):
        """Return the rate of `y`, a reduction over the sets of neighbours of the member."""
        over = RadiusNeighborhood(self.radius)
        terms = {
            'flank': lambda: hyperedges(self._flank(s), size=3, over=over),
            'count': lambda: hyperedges(SR(1), reduce='count', size=4, over=over),
            'pairs': lambda: hyperedges(SR(1), size=3, over=over),
        }
        return [RateLaw(s.y, terms[self.term]())]

    @staticmethod
    def _flank(s):
        """Return the flanking term of one pair of neighbours, half of one minus the cosine of their angle."""
        first, second = s.others[0].x - s.x, s.others[1].x - s.x
        return (1 - first * second / sqrt((first * first) * (second * second))) / 2


def _neighbour_set_drift(positions, term, side, backend, radius=1.5):
    """Return the drift of `y` of members at positions under a reduction over neighbour hyperedges."""
    space = ContinuousSpace(extent=(side, side), boundary='periodic')
    values = {'x': positions, 'term': term, 'radius': radius}
    group = GroupElement('g', element=NeighbourSetElement, size=len(positions), values=values)
    problem = PassCompiler(backend=backend).prepare(System(elements=[group], space=space)).run().problem_
    return problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'g.y')


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_neighbour_hyperedges_match_a_loop_over_pairs_of_neighbours(backend):
    """Test that a three-body term over pairs of neighbours equals a loop over every member and pair."""
    side, radius = 5.0, 1.5
    positions, _, _ = _scattered(30, side, 7)
    difference = _minimum_image(positions[None, :, :] - positions[:, None, :], side)
    distance = np.linalg.norm(difference, axis=2)
    expected = np.zeros(30)
    for member in range(30):
        around = [other for other in range(30) if other != member and distance[member, other] <= radius]
        for first, second in itertools.combinations(around, 2):
            one, two = difference[member, first], difference[member, second]
            expected[member] += (1 - one @ two / (np.linalg.norm(one) * np.linalg.norm(two))) / 2
    drift = _neighbour_set_drift(positions, 'flank', side, backend, radius)
    assert np.abs(expected).sum() > 0
    assert np.allclose(drift, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_neighbour_hyperedges_of_four_members_are_the_triples_of_neighbours(backend):
    """Test that a member with four neighbours has four hyperedges of size four."""
    positions = np.array([[5.0, 5.0], [5.5, 5.0], [4.5, 5.0], [5.0, 5.5], [5.0, 4.5]])
    drift = _neighbour_set_drift(positions, 'count', 10.0, backend, radius=0.6)
    assert drift.tolist() == [4.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize('backend', BACKENDS, ids=['numpy', 'numba'])
def test_neighbour_hyperedges_need_enough_neighbours(backend):
    """Test that a member with one neighbour has a zero sum over hyperedges of size three."""
    positions = np.array([[1.0, 1.0], [1.5, 1.0], [8.0, 8.0]])
    drift = _neighbour_set_drift(positions, 'pairs', 10.0, backend, radius=1.0)
    assert drift.tolist() == [0.0, 0.0, 0.0]


def test_validate_reports_neighbour_hyperedges_on_a_network():
    """Test that a reduction over neighbour hyperedges in a network space has a law-space issue."""
    group = GroupElement('g', element=NeighbourSetElement, size=5, values={'term': 'pairs'})
    system = System(elements=[group], space=NetworkSpace(nx.cycle_graph(5)))
    issues = [issue for issue in PassCompiler().prepare(system).report_.issues if issue.code == 'law-space']
    assert len(issues) == 1
    assert 'continuous space' in issues[0].message


def test_validate_reports_neighbour_hyperedges_on_the_jax_backend():
    """Test that a reduction over neighbour hyperedges on the JAX backend names the backends that run it."""
    positions = np.array([[1.0, 1.0], [1.5, 1.0], [1.0, 1.5]])
    group = GroupElement('g', element=NeighbourSetElement, size=3, values={'x': positions, 'term': 'pairs'})
    system = System(elements=[group], space=ContinuousSpace(extent=(4.0, 4.0)))
    issues = PassCompiler(backend=JaxBackend()).prepare(system).report_.issues
    (issue,) = [issue for issue in issues if issue.code == 'backend-unsupported']
    assert 'NumpyBackend' in issue.fix
    assert 'NumbaBackend' in issue.fix


class QuenchedEnergyElement(Element):
    """Member with a quenched energy."""

    def __init__(self, label: str, energy: Annotated[float, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.energy = energy


class LogPartitionElement(Element):
    """Element holding the reduction of a group."""

    def __init__(self, label: str, value: Annotated[float, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.value = value


class LogPartitionInteraction(Interaction):
    """Interaction that equates a state to the log-sum-exp of a group of energies."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        energies: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.energies = energies

    def _get_laws(self, s: SimpleNamespace) -> list[BasePart]:
        """Return the equation of the reduction."""
        return [EquationLaw(s.target.value == log_sum_exp(s.energies.energy, over=s.energies), target=s.target.value)]


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_log_sum_exp_large_values(backend: type) -> None:
    """Test that values of order 10^4 over 1000 members give the largest value plus at most `ln 1000`."""
    values = np.random.default_rng(0).uniform(1e4, 2e4, 1000)
    group = GroupElement('energies', element=QuenchedEnergyElement, size=1000, values={'energy': values})
    reader = LogPartitionInteraction('reader', target='reduced', energies='energies')
    system = System(elements=[group, LogPartitionElement('reduced')], interactions=[reader], time=StaticTime())
    trajectory = simulate(system, until=1.0, backend=backend())
    result = trajectory.values('reduced.value')[0]
    assert np.isfinite(result)
    assert values.max() <= result <= values.max() + math.log(1000)
    assert result == pytest.approx(scipy.special.logsumexp(values))


class BoltzmannElement(Element):
    """Element holding the log partition function, the mean energy, and the entropy per spin of a group of levels."""

    def __init__(
        self,
        label: str,
        log_partition: Annotated[float, StateSpec()] = 0.0,
        energy: Annotated[float, StateSpec()] = 0.0,
        entropy: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.log_partition = log_partition
        self.energy = energy
        self.entropy = entropy


class BoltzmannInteraction(Interaction):
    """Interaction that equates the states of a Boltzmann element to their sums over a group of levels."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        levels: Annotated[str, RoleSpec()],
        n_spins: Annotated[float, ParameterSpec()] = 20.0,
        temperature: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.levels = levels
        self.n_spins = n_spins
        self.temperature = temperature

    def _get_laws(self, s: SimpleNamespace) -> list[BasePart]:
        """Return the equations of the log partition function, the mean energy, and the entropy per spin."""
        weight = -s.levels.energy / s.temperature
        log_partition = log_sum_exp(weight, over=s.levels)
        energy = total(s.levels.energy * exp(weight - log_partition), over=s.levels)
        entropy = (s.target.log_partition + s.target.energy / s.temperature) / s.n_spins
        return [
            EquationLaw(s.target.log_partition == log_partition, target=s.target.log_partition),
            EquationLaw(s.target.energy == energy, target=s.target.energy),
            EquationLaw(s.target.entropy == entropy, target=s.target.entropy),
        ]


def _random_levels(temperature: float = 1.0, random_state: int = 0) -> System:
    """Return 2^20 quenched Gaussian energies of variance N/2 and their Boltzmann sums at a temperature."""
    n_spins = 20
    energies = np.random.default_rng(random_state).normal(0.0, np.sqrt(n_spins / 2), 2**n_spins)
    levels = GroupElement('levels', element=QuenchedEnergyElement, size=2**n_spins, values={'energy': energies})
    reader = BoltzmannInteraction('reader', target='sums', levels='levels', n_spins=n_spins, temperature=temperature)
    return System(elements=[levels, BoltzmannElement('sums')], interactions=[reader], time=StaticTime())


@pytest.mark.slow
def test_log_sum_exp_random_energy_system_entropy() -> None:
    """Test that the entropy of random levels over 10 seeds follows the theory above the glass temperature."""
    bounds = {'temperature': [1.0, 2.0, 0.3], 'random_state': list(range(10))}
    result = sweep(_random_levels, bounds, until=1.0, outputs=['sums.entropy'])
    entropies = result.series('sums.entropy')[:, 0].reshape(3, 10).mean(axis=1)
    assert entropies[0] == pytest.approx(math.log(2) - 1 / 4, abs=0.02)
    assert entropies[1] == pytest.approx(math.log(2) - 1 / 16, abs=0.02)
    assert entropies[2] < 0.1
