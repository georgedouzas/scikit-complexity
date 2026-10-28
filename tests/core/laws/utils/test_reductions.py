"""Test the `_reductions` module of `skcomplex.core.laws.utils`."""

import math
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
import scipy.sparse
from sage.all__sagemath_symbolics import RR, atan2, cos, pi, sin, tanh, var, vector

from skcomplex.core.elements import Element, GroupElement, LatticeElement
from skcomplex.core.errors import ValidationError
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import RateLaw, UpdateLaw
from skcomplex.core.laws.utils import (
    NEIGHBOR_REDUCTIONS,
    count,
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
from skcomplex.core.spaces import (
    ConeNeighborhood,
    ContinuousSpace,
    DiscreteTime,
    LatticeSpace,
    NearestNeighborhood,
    NetworkSpace,
    RadiusNeighborhood,
    VonNeumannNeighborhood,
)
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.compile import PassCompiler, compile_system
from skcomplex.engine.problems._compiled_reduction import CompiledReduction
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver


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
    assert CompiledReduction._reduce_values([1.0, 2.0], 'count') == 2.0
    assert CompiledReduction._reduce_values([], 'count') == 0.0


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
        x: Annotated[object, StateSpec(vector=True, position=True)] = 0.0,
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
        x: Annotated[object, StateSpec(vector=True, position=True)] = 0.0,
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
    }
    return reductions[kind]


def _outputs(system, backend):
    """Return the values of `out` after one update of the first group of a system."""
    problem = compile_system(system, backend=backend)
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
    over = ConeNeighborhood(2.0, math.pi / 3, heading='phi')
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
    over = ConeNeighborhood(radius, angle, heading='phi')
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
            x: Annotated[object, StateSpec(vector=True, position=True)] = 0.0,
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
    problem = compile_system(system)
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
        compile_system(system)


def test_validate_reports_a_metric_neighbourhood_on_a_network():
    """Test that a law reducing over a radius in a system on a network space has a law-space issue."""
    system = System(
        elements=[GroupElement('g', element=ReducingElement, size=5, values={'over': RadiusNeighborhood(1.0)})],
        space=NetworkSpace(nx.cycle_graph(5)),
        time=DiscreteTime(),
    )
    issues = [issue for issue in PassCompiler().check().prepare(system).report_.issues if issue.code == 'law-space']
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
    assert [issue.code for issue in PassCompiler().check().prepare(system).report_.issues] == ['law-space']


def test_validate_reports_a_cone_without_a_heading():
    """Test that a cone over members without a velocity or heading state has a law-domain issue."""
    over = ConeNeighborhood(1.0, 1.0)
    system = System(
        elements=[GroupElement('g', element=PositionedReducingElement, size=5, values={'over': over})],
        space=ContinuousSpace(extent=(2.0, 2.0)),
        time=DiscreteTime(),
    )
    issues = PassCompiler().check().prepare(system).report_.issues
    assert [issue.code for issue in issues] == ['law-domain']
    assert 'velocity_of' in issues[0].fix


def test_validate_accepts_a_cone_along_the_velocity():
    """Test that a cone reads the vector velocity of the position as the heading of every member."""

    class MovingElement(Element):
        """Moving element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            out: Annotated[object, StateSpec()] = 0.0,
            y: Annotated[object, StateSpec()] = 0.0,
            x: Annotated[object, StateSpec(vector=True, position=True)] = 0.0,
            v: Annotated[object, StateSpec(vector=True, velocity_of='x')] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.out = out
            self.y = y
            self.x = x
            self.v = v

        def _get_laws(self, s):
            """Return the laws of the moving element."""
            return [RateLaw(s.out, neighbors(s.other.y, over=ConeNeighborhood(1.0, 1.0)))]

    moving = MovingElement
    system = System(elements=[GroupElement('g', element=moving, size=5)], space=ContinuousSpace(extent=(2.0, 2.0)))
    assert PassCompiler().check().prepare(system).report_.ok


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
    below = (
        Simulation(_kuramoto(0.5 * CRITICAL), dt=0.5, solver=RK4Solver(dt=0.1), backend=NumbaBackend())
        .check()
        .run(10.0)
    )
    above = (
        Simulation(_kuramoto(3 * CRITICAL), dt=0.5, solver=RK4Solver(dt=0.1), backend=NumbaBackend())
        .check()
        .run(10.0)
    )
    weak, strong = _order(below.trajectory_.values('oscillators.theta')), _order(
        above.trajectory_.values('oscillators.theta'),
    )
    assert weak[-5:].max() < 0.1
    assert strong[-5:].min() > 0.7
    assert strong[-1] > 10 * strong[0]


def test_kuramoto_agrees_across_backends():
    """Test that the Kuramoto run on a complete graph gives the same phases on NumPy and Numba."""
    runs = [
        (
            Simulation(_kuramoto(3 * CRITICAL), dt=0.5, solver=RK4Solver(dt=0.1), backend=backend)
            .check()
            .run(2.0)
            .trajectory_
        )
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
        x: Annotated[object, StateSpec(vector=True, position=True)] = 0.0,
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
    """Return the Vicsek model at high density in a periodic box under discrete time."""
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
    """Test that the polar order parameter of the Vicsek model exceeds 0.9 at low noise and high density."""
    trajectory = Simulation(_vicsek(), dt=1.0, random_state=0, backend=NumbaBackend()).check().run(100).trajectory_
    order = _order(trajectory.values('movers.theta'))
    assert order[0] < 0.2
    assert order[-1] > 0.9


def test_vicsek_agrees_across_backends():
    """Test that the Vicsek model gives the same headings and positions on NumPy and Numba for one seed."""
    runs = [Simulation(
        _vicsek(), dt=1.0, random_state=0, backend=backend,
    ).check().run(50).trajectory_ for backend in BACKENDS]
    turns = np.angle(np.exp(1j * (runs[0].values('movers.theta') - runs[1].values('movers.theta'))))
    assert np.abs(turns).max() < 1e-12
    assert np.allclose(runs[0].values('movers.x'), runs[1].values('movers.x'), rtol=0, atol=1e-12)
