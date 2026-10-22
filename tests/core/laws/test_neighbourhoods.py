"""Test reductions over neighbourhoods: networks, lattices, radii, nearest members, and cones, on NumPy and Numba."""

import math

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import atan2, cos, sin

from skcomplex.core.base import Domain, ValidationError, state, symbol, vector
from skcomplex.core.compile import compile_system, validate
from skcomplex.core.components import Element
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import NEIGHBOUR_REDUCTIONS, Rate, Update, neighbors, neighbourhoods_in, uniform
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.space import (
    Cone,
    ContinuousSpace,
    DiscreteTime,
    LatticeSpace,
    Nearest,
    Neighborhood,
    NetworkSpace,
    Radius,
)
from skcomplex.core.system import System

BACKENDS = ('numpy', 'numba')
GAMMA = 0.5
CRITICAL = 2 * GAMMA


def _reducing(over, kind, *, weight=None, **states):
    """Return an element type whose state `out` is the reduction of the neighbour's `y` over a neighbourhood."""
    return Element.define(
        'Reducing',
        states={'out': 0.0, 'y': 0.0, **states},
        laws=lambda s: [Update(s.out, neighbors(s.other.y, over=over, reduce=kind, weight=weight))],
    )


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
    return problem.states.read(after, f'{system.components[0].label}.out')


def _minimum_image(difference, side):
    """Return displacements wrapped to the minimum image in a periodic box of one side."""
    return difference - side * np.rint(difference / side)


def test_neighbors_keeps_the_values_at_lattice_offsets():
    """Test that a lattice neighbourhood without a reduction gives the values at every offset, for local energies."""
    values = neighbors(symbol('spin'), Neighborhood.von_neumann())
    assert [str(item) for item in values] == [f'neighbor(spin, {index})' for index in range(4)]


def test_neighbors_rejects_an_unknown_reduction():
    """Test that an unknown reduction raises an error that lists the reductions."""
    with pytest.raises(ValidationError, match='normalized-mean'):
        neighbors(1, reduce='median')


def test_neighbourhoods_in_finds_the_neighbourhood_and_the_weight():
    """Test that a reduction found in an expression keeps its neighbourhood, kind, and weight."""
    over = Cone(2.0, math.pi / 3, heading='phi')
    found = neighbourhoods_in(1 + neighbors(2, over=over, reduce='normalized-mean', weight=3))
    assert [(item.over, item.kind, item.weight) for item in found] == [(over, 'normalized-mean', 3)]


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('kind', NEIGHBOUR_REDUCTIONS)
def test_reduction_over_a_network_matches_its_definition(kind, backend):
    """Test that every reduction over the neighbours of a network node matches the reduction of their values."""
    graph = nx.gnp_random_graph(12, 0.3, seed=1)
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    values = np.random.default_rng(0).choice([-2.0, 0.0, 1.0, 3.5], 12)
    system = System(components=[_reducing(None, kind).group('g', 12, y=values)], space=NetworkSpace(graph))
    expected = [_expected(kind, values[list(graph.neighbors(node))], np.ones(graph.degree(node))) for node in graph]
    assert np.allclose(_outputs(system, backend), expected, atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS)
def test_weighted_reduction_reads_the_member_and_its_neighbour(backend):
    """Test that a weight of the member and the neighbour weighs every neighbour in a normalized mean."""
    weighted = Element.define(
        'Weighted',
        states={'out': 0.0, 'y': 0.0},
        laws=lambda s: [Update(s.out, neighbors(s.other.y, reduce='normalized-mean', weight=s.y + s.other.y))],
    )
    values = np.array([1.0, 2.0, 3.0, 4.0])
    system = System(components=[weighted.group('g', 4, y=values)], space=NetworkSpace(nx.path_graph(4)))
    expected = [
        _expected('normalized-mean', values[others], values[member] + values[others])
        for member, others in enumerate([[1], [0, 2], [1, 3], [2]])
    ]
    assert np.allclose(_outputs(system, backend), expected, atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS)
def test_reduction_over_a_lattice_leaves_out_sites_outside_fixed_edges(backend):
    """Test that a sum over a von Neumann neighbourhood of a fixed lattice sums the neighbours inside the lattice."""
    values = np.arange(12.0).reshape(3, 4)
    reducing = _reducing(Neighborhood.von_neumann(), 'sum')
    space = LatticeSpace((3, 4), boundary='fixed')
    system = System(components=[reducing.lattice('g', space, y=values)])
    padded = np.pad(values, 1)
    expected = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    assert np.allclose(_outputs(system, backend), expected.ravel(), atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS)
def test_reduction_without_a_neighbourhood_reads_the_lattice_neighbourhood(backend):
    """Test that a reduction without a neighbourhood on a periodic lattice reads the lattice's Moore neighbourhood."""
    values = np.random.default_rng(2).normal(size=(4, 5))
    system = System(components=[_reducing(None, 'sum').lattice('g', LatticeSpace((4, 5)), y=values)])
    expected = sum(np.roll(values, (a - 1, b - 1), axis=(0, 1)) for a, b in np.ndindex(3, 3)) - values
    assert np.allclose(_outputs(system, backend), expected.ravel(), atol=1e-12)


def _scattered(count, side, seed):
    """Return positions, values, and heading angles of members scattered in a square."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, side, (count, 2)), rng.normal(size=count), rng.uniform(-math.pi, math.pi, count)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('boundary', ['periodic', 'open'])
@pytest.mark.parametrize('center', [False, True])
def test_reduction_within_a_radius_matches_a_direct_search(boundary, center, backend):
    """Test that a sum over the members within a radius matches a search of every pair, with minimum images."""
    side, radius = 4.0, 0.9
    positions, values, _ = _scattered(80, side, 3)
    reducing = _reducing(Radius(radius, center=center), 'sum', x=state(vector=True, position=True))
    space = ContinuousSpace(side, side, boundary=boundary)
    system = System(components=[reducing.group('g', 80, x=positions, y=values)], space=space)
    difference = positions[:, None, :] - positions[None, :, :]
    if boundary == 'periodic':
        difference = _minimum_image(difference, side)
    within = np.linalg.norm(difference, axis=2) <= radius
    np.fill_diagonal(within, center)
    assert np.allclose(_outputs(system, backend), within @ values, atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('boundary', ['periodic', 'open'])
def test_reduction_over_the_nearest_members_matches_a_direct_search(boundary, backend):
    """Test that a mean over the k nearest members matches a sort of every distance."""
    side, k = 5.0, 6
    positions, values, _ = _scattered(70, side, 4)
    reducing = _reducing(Nearest(k), 'mean', x=state(vector=True, position=True))
    space = ContinuousSpace(side, side, boundary=boundary)
    system = System(components=[reducing.group('g', 70, x=positions, y=values)], space=space)
    difference = positions[:, None, :] - positions[None, :, :]
    if boundary == 'periodic':
        difference = _minimum_image(difference, side)
    distances = np.linalg.norm(difference, axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1)[:, :k]
    assert np.allclose(_outputs(system, backend), values[nearest].mean(axis=1), atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS)
def test_reduction_inside_a_cone_keeps_the_members_ahead(backend):
    """Test that a count over a cone of a heading angle counts the members within the radius and the half-angle."""
    side, radius, angle = 4.0, 1.2, math.pi / 3
    positions, values, headings = _scattered(90, side, 5)
    reducing = _reducing(
        Cone(radius, angle, heading='phi'),
        'count',
        x=state(vector=True, position=True),
        phi=state(0.0, domain=Domain.ANGLE),
    )
    space = ContinuousSpace(side, side, boundary='periodic')
    system = System(components=[reducing.group('g', 90, x=positions, y=values, phi=headings)], space=space)
    direction = -_minimum_image(positions[:, None, :] - positions[None, :, :], side)
    distances = np.linalg.norm(direction, axis=2)
    heading = np.stack([np.cos(headings), np.sin(headings)], axis=1)
    cosines = np.einsum('ik,ijk->ij', heading, direction) / np.where(distances > 0, distances, 1)
    inside = (distances <= radius) & (cosines >= math.cos(angle))
    np.fill_diagonal(inside, False)
    assert np.allclose(_outputs(system, backend), inside.sum(axis=1), atol=1e-12)


@pytest.mark.parametrize('backend', BACKENDS)
def test_neighbour_positions_are_read_by_minimum_image(backend):
    """Test that the position of a neighbour across a periodic edge is its image nearest to the member."""
    reading = Element.define(
        'Reading',
        states={'out': 0.0, 'x': state(vector=True, position=True)},
        laws=lambda s: [Update(s.out, neighbors(s.other.x[0] - s.x[0], over=Radius(1.0)))],
    )
    positions = np.array([[0.2, 0.5], [9.9, 0.5]])
    space = ContinuousSpace(10.0, 10.0, boundary='periodic')
    system = System(components=[reading.group('g', 2, x=positions)], space=space)
    assert np.allclose(_outputs(system, backend), [-0.3, 0.3], atol=1e-12)


def test_interaction_reduces_over_the_neighbours_of_its_group():
    """Test that an interaction reads the neighbours of the group bound to its role through `s.target.other`."""
    diffusion = Interaction.define(
        'Diffusion',
        roles=['target'],
        laws=lambda s: [Rate(s.target.y, neighbors(s.target.other.y - s.target.y))],
    )
    plain = Element.define('Plain', states={'y': 0.0})
    system = System(
        components=[plain.group('g', 3, y=[0.0, 1.0, 4.0])],
        interactions=[diffusion('d', target='g')],
        space=NetworkSpace(nx.path_graph(3)),
    )
    problem = compile_system(system)
    assert problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist() == [1.0, 2.0, -3.0]


def test_neighbour_symbols_outside_a_reduction_are_rejected():
    """Test that a law reading `s.other` outside a neighbourhood reduction raises an error that names `neighbors`."""
    stray = Element.define('Stray', states={'y': 0.0}, laws=lambda s: [Rate(s.y, s.other.y)])
    system = System(components=[stray.group('g', 3)], space=NetworkSpace(nx.path_graph(3)))
    with pytest.raises(ValidationError, match='neighbors'):
        compile_system(system)


def test_validate_reports_a_metric_neighbourhood_on_a_network():
    """Test that a law reducing over a radius in a system on a network space has a law-space issue."""
    system = System(
        components=[_reducing(Radius(1.0), 'sum').group('g', 5)],
        space=NetworkSpace(nx.cycle_graph(5)),
        time=DiscreteTime(),
    )
    issues = [issue for issue in validate(system).issues if issue.code == 'law-space']
    assert len(issues) == 1
    assert 'continuous space' in issues[0].fix
    assert 'over=None' in issues[0].fix


def test_validate_reports_a_network_neighbourhood_in_a_continuous_space():
    """Test that a reduction over the neighbours of the space in a continuous space has a law-space issue."""
    reducing = _reducing(None, 'sum', x=state(vector=True, position=True))
    system = System(components=[reducing.group('g', 5)], space=ContinuousSpace(2.0, 2.0), time=DiscreteTime())
    assert [issue.code for issue in validate(system).issues] == ['law-space']


def test_validate_reports_a_cone_without_a_heading():
    """Test that a cone over members without a velocity or heading state has a law-domain issue."""
    reducing = _reducing(Cone(1.0, 1.0), 'sum', x=state(vector=True, position=True))
    system = System(components=[reducing.group('g', 5)], space=ContinuousSpace(2.0, 2.0), time=DiscreteTime())
    issues = validate(system).issues
    assert [issue.code for issue in issues] == ['law-domain']
    assert 'velocity_of' in issues[0].fix


def test_validate_accepts_a_cone_along_the_velocity():
    """Test that a cone reads the vector velocity of the position as the heading of every member."""
    moving = Element.define(
        'Moving',
        states={
            'out': 0.0,
            'y': 0.0,
            'x': state(vector=True, position=True),
            'v': state(vector=True, velocity_of='x'),
        },
        laws=lambda s: [Rate(s.out, neighbors(s.other.y, over=Cone(1.0, 1.0)))],
    )
    system = System(components=[moving.group('g', 5)], space=ContinuousSpace(2.0, 2.0))
    assert validate(system).ok


Oscillator = Element.define(
    'Oscillator',
    states={'theta': 0.0},
    parameters={'omega': 0.0, 'K': 1.0},
    laws=lambda s: [Rate(s.theta, s.omega + s.K * neighbors(sin(s.other.theta - s.theta), reduce='mean'))],
)


def _kuramoto(coupling, size=1000):
    """Return oscillators with Lorentzian frequencies of half-width `GAMMA` on a complete graph."""
    quantiles = (np.arange(size) + 0.5) / size
    omega = GAMMA * np.tan(math.pi * (quantiles - 0.5))
    theta = np.random.default_rng(1).uniform(0, 2 * math.pi, size)
    oscillators = Oscillator.group('oscillators', size, theta=theta, omega=omega, K=coupling)
    return System(components=[oscillators], space=NetworkSpace(nx.complete_graph(size)))


def _order(phases):
    """Return the modulus of the mean phase vector of every frame."""
    return np.abs(np.exp(1j * phases).mean(axis=1))


def test_kuramoto_synchronizes_above_the_critical_coupling():
    """Test that 1000 oscillators on a complete graph stay incoherent below `2 gamma` and synchronize above it."""
    below = Simulation(_kuramoto(0.5 * CRITICAL), dt=0.5, solver=RK4(dt=0.1), backend='numba').run(10.0)
    above = Simulation(_kuramoto(3 * CRITICAL), dt=0.5, solver=RK4(dt=0.1), backend='numba').run(10.0)
    weak, strong = _order(below.trajectory_['oscillators.theta']), _order(above.trajectory_['oscillators.theta'])
    assert weak[-5:].max() < 0.1
    assert strong[-5:].min() > 0.7
    assert strong[-1] > 10 * strong[0]


def test_kuramoto_agrees_across_backends():
    """Test that the Kuramoto run on a complete graph gives the same phases on NumPy and Numba."""
    runs = [
        Simulation(_kuramoto(3 * CRITICAL), dt=0.5, solver=RK4(dt=0.1), backend=backend).run(2.0).trajectory_
        for backend in BACKENDS
    ]
    assert np.allclose(runs[0]['oscillators.theta'], runs[1]['oscillators.theta'], rtol=0, atol=1e-12)


RADIUS = 1.0
Mover = Element.define(
    'Mover',
    states={'x': state(vector=True, position=True), 'theta': state(0.0, domain=Domain.ANGLE)},
    parameters={'v0': 0.03, 'eta': 0.1},
    laws=lambda s: [
        Update(
            s.theta,
            atan2(
                neighbors(sin(s.other.theta), over=Radius(RADIUS, center=True)),
                neighbors(cos(s.other.theta), over=Radius(RADIUS, center=True)),
            )
            + uniform('noise', -s.eta / 2, s.eta / 2),
        ),
        Update(s.x, s.x + s.v0 * vector([cos(s.theta), sin(s.theta)])),
    ],
)


def _vicsek(size=400, side=5.0):
    """Return the Vicsek model at high density in a periodic box under discrete time."""
    rng = np.random.default_rng(3)
    movers = Mover.group('movers', size, x=rng.uniform(0, side, (size, 2)), theta=rng.uniform(-math.pi, math.pi, size))
    return System(components=[movers], space=ContinuousSpace(side, side, boundary='periodic'), time=DiscreteTime())


def test_vicsek_flocks_at_low_noise_and_high_density():
    """Test that the polar order parameter of the Vicsek model exceeds 0.9 at low noise and high density."""
    trajectory = Simulation(_vicsek(), dt=1.0, random_state=0, backend='numba').run(100).trajectory_
    order = _order(trajectory['movers.theta'])
    assert order[0] < 0.2
    assert order[-1] > 0.9


def test_vicsek_agrees_across_backends():
    """Test that the Vicsek model gives the same headings and positions on NumPy and Numba for one seed."""
    runs = [Simulation(_vicsek(), dt=1.0, random_state=0, backend=backend).run(50).trajectory_ for backend in BACKENDS]
    turns = np.angle(np.exp(1j * (runs[0]['movers.theta'] - runs[1]['movers.theta'])))
    assert np.abs(turns).max() < 1e-12
    assert np.allclose(runs[0]['movers.x'], runs[1]['movers.x'], rtol=0, atol=1e-12)
