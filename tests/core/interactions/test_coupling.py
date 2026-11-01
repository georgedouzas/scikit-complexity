"""Test the `_coupling` module of `skcomplex.core.interactions`."""

import math
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
import scipy.integrate
import scipy.optimize
from sage.all__sagemath_symbolics import SR, I, atan2, cos, exp, pi, sin, vector

from skcomplex.core.elements import Element, GroupElement, LatticeElement
from skcomplex.core.interactions import CouplingInteraction
from skcomplex.core.laws import BaseLaw, RateLaw, UpdateLaw
from skcomplex.core.laws.utils import count, uniform, where
from skcomplex.core.spaces import ContinuousSpace, DiscreteTime, LatticeSpace, NetworkSpace, RadiusNeighborhood
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.errors import ValidationError
from skcomplex.specs import ClampSpec, ObservableSpec, ParameterSpec, RadiusSpec, RoleSpec, StateSpec, VisualSpec

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])
GAMMA = 0.5


class OscillatorElement(Element):
    """Oscillator with a phase and a natural frequency."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: Annotated[object, StateSpec()] = 0.0,
        omega: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, s):
        """Return the rotation at the natural frequency."""
        return [RateLaw(s.theta, s.omega)]


class CellElement(Element):
    """Cell with a level and no dynamics of its own."""

    def __init__(self, label: str, visual: VisualSpec | None = None, *, u: Annotated[object, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.u = u


class MoverElement(Element):
    """Member that moves at a constant speed along its heading, every step of discrete time."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec(vector=True, position=True)] = 0.0,
        theta: Annotated[object, StateSpec(period=2 * pi)] = 0.0,
        v0: Annotated[object, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.theta = theta
        self.v0 = v0

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the move along the heading."""
        return [UpdateLaw(s.x, s.x + s.v0 * vector(SR, [cos(s.theta), sin(s.theta)]))]


class LinkElement(Element):
    """Link of a network with a weight."""

    def __init__(self, label: str, visual: VisualSpec | None = None, *, w: Annotated[object, StateSpec()] = 1.0):
        super().__init__(label, visual=visual)
        self.w = w


class WeightedCouplingInteraction(CouplingInteraction):
    """Coupling along the edges of a network, weighted by the state `w` of the group on its edges."""

    def __init__(self, label: str, *, target: Annotated[str, RoleSpec()], links: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, target=target, read='theta', write='theta', function=_sine, weights='w')
        self.links = links


def _sine(s):
    """Return the sine of the phase difference."""
    return sin(s.other.theta - s.member.theta)


def _oscillators(space, theta, omega, **arguments):
    """Return oscillators on a space coupled by a function, the sine of the phase difference by default."""
    oscillators = LatticeElement(
        'oscillators',
        element=OscillatorElement,
        space=space,
        values={'theta': theta, 'omega': omega},
    )
    coupling = CouplingInteraction('coupling', target='oscillators', read='theta', write='theta', **arguments)
    return System(elements=[oscillators], interactions=[coupling])


def _phases(system, until, backend=None):
    """Return the phases of the oscillators at every unit of time."""
    simulation = Simulation(system, dt=1.0, solver=RK4Solver(dt=0.05), backend=backend).check().prepare()
    return simulation.run(until).trajectory_.values('oscillators.theta')


def _drift(system, path='oscillators.theta', backend=None):
    """Return the rates of a state at the initial state."""
    problem = compile_system(system, backend=backend)
    return problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), path)


def _order(phases):
    """Return the modulus of the mean phase vector of every frame."""
    return np.abs(np.exp(1j * phases).mean(axis=1))


def _group(theta, **arguments):
    """Return oscillators in a group without a space, coupled by the sine of the phase difference."""
    group = GroupElement('oscillators', element=OscillatorElement, size=len(theta), values={'theta': theta})
    coupling = CouplingInteraction('coupling', target='oscillators', read='theta', write='theta', **arguments)
    return System(elements=[group], interactions=[coupling])


@BACKENDS
@pytest.mark.parametrize('mode', ['edges', 'group'])
def test_coupling_synchronizes_kuramoto_oscillators_above_the_critical_strength(mode, backend):
    """Test that 500 oscillators on a complete graph coupled above the critical strength reach an order above 0.8."""
    size = 500
    omega = GAMMA * np.tan(math.pi * ((np.arange(size) + 0.5) / size - 0.5))
    theta = np.random.default_rng(1).uniform(0, 2 * math.pi, size)
    strength = 8 * GAMMA
    system = _oscillators(
        NetworkSpace(nx.complete_graph(size)),
        theta,
        omega,
        function=lambda s: strength * _sine(s),
        reduce='mean',
        mode=mode,
        include_self=True,
    )
    order = _order(_phases(system, 20.0, backend))
    assert order[0] < 0.1
    assert order[-5:].min() > 0.8


@BACKENDS
def test_coupling_by_difference_diffuses_along_the_bonds(backend):
    """Test that the default coupling adds `x_j - x_i` for every neighbour along both directions of a bond."""
    theta = np.array([1.0, 0.0, 0.0, 3.0])
    system = _oscillators(LatticeSpace(4), theta, np.zeros(4), function=lambda s: 2 * (s.other.theta - s.member.theta))
    assert _drift(system, backend=backend).tolist() == [2.0, 2.0, 6.0, -10.0]
    assert _drift(_oscillators(LatticeSpace(4), theta, np.zeros(4)), backend=backend).tolist() == [1.0, 1.0, 3.0, -5.0]


@BACKENDS
def test_coupling_mean_over_edges_includes_the_member_itself(backend):
    """Test that the mean over the edges with the member itself divides the sum by the degree plus one."""
    theta = np.array([0.0, 1.0, 2.5, 4.0])
    system = _oscillators(LatticeSpace(4), theta, np.zeros(4), reduce='mean', include_self=True)
    expected = (np.roll(theta, 1) + np.roll(theta, -1) - 2 * theta) / 3
    np.testing.assert_allclose(_drift(system, backend=backend), expected, atol=1e-12)


@BACKENDS
def test_coupling_over_the_group_excludes_the_member_unless_included(backend):
    """Test that the group coupling is the mean of the coupling to every other member, or to every member."""
    theta = np.random.default_rng(2).uniform(0, 2 * math.pi, 6)
    differences = np.sin(theta[None, :] - theta[:, None] - 0.3)
    excluded = _group(theta, function=_sine, lag=0.3, mode='group', reduce='mean')
    included = _group(theta, function=_sine, lag=0.3, mode='group', reduce='mean', include_self=True)
    np.testing.assert_allclose(_drift(excluded, backend=backend), (differences.sum(1) - math.sin(-0.3)) / 5)
    np.testing.assert_allclose(_drift(included, backend=backend), differences.mean(axis=1), atol=1e-12)


@BACKENDS
def test_coupling_over_the_group_weights_every_member_by_a_matrix(backend):
    """Test that a weight matrix over the group gives `sum_j W_ij f(x_j)` with a zero diagonal."""
    theta = np.array([0.0, 1.0, 2.5, 4.0])
    weights = np.arange(16.0).reshape(4, 4)
    system = _group(theta, function=lambda s: sin(s.other.theta), mode='group', weights=weights)
    expected = (weights - np.diag(np.diag(weights))) @ np.sin(theta)
    np.testing.assert_allclose(_drift(system, backend=backend), expected, atol=1e-12)


def _two_groups(theta, phi, **arguments):
    """Return two groups of oscillators, the first coupled to the members of the second."""
    first = GroupElement('first', element=OscillatorElement, size=len(theta), values={'theta': theta})
    second = GroupElement('second', element=OscillatorElement, size=len(phi), values={'theta': phi})
    coupling = CouplingInteraction(
        'coupling',
        target='first',
        source='second',
        read='theta',
        write='theta',
        mode='group',
        **arguments,
    )
    return System(elements=[first, second], interactions=[coupling])


@BACKENDS
def test_coupling_over_a_source_group_reduces_over_its_every_member(backend):
    """Test that a source group gives every member of the target the mean over every member of the source."""
    theta, phi = np.array([0.0, 1.0, 2.0]), np.array([0.5, 3.0, -1.0, 2.2])
    system = _two_groups(theta, phi, function=_sine, lag=0.3, reduce='mean')
    expected = np.sin(phi[None, :] - theta[:, None] - 0.3).mean(axis=1)
    np.testing.assert_allclose(_drift(system, 'first.theta', backend), expected, atol=1e-12)
    np.testing.assert_allclose(_drift(system, 'second.theta', backend), 0.0)


@BACKENDS
@pytest.mark.parametrize('reduce', ['max', 'min'])
def test_coupling_over_a_source_group_takes_the_largest_or_smallest_term(reduce, backend):
    """Test that the largest and smallest reductions take the extreme weighted term over the source group."""
    theta, phi = np.array([0.0, 1.0, 2.0]), np.array([0.5, 3.0, -1.0, 2.2])
    system = _two_groups(theta, phi, function=_sine, reduce=reduce, weights=lambda s: 2.0)
    differences = 2.0 * np.sin(phi[None, :] - theta[:, None])
    expected = differences.max(axis=1) if reduce == 'max' else differences.min(axis=1)
    np.testing.assert_allclose(_drift(system, 'first.theta', backend), expected, atol=1e-12)


@BACKENDS
@pytest.mark.parametrize('mode', ['group', 'edges'])
def test_coupling_normalized_mean_weights_members_by_their_states(mode, backend):
    """Test that the normalized mean with indicator weights averages the members within a distance, itself included."""
    theta = np.array([0.0, 0.3, 0.5, 2.0, 2.4])
    system = _oscillators(
        NetworkSpace(nx.complete_graph(len(theta))),
        theta,
        np.zeros(len(theta)),
        function=lambda s: s.other.theta,
        weights=lambda s: where(abs(s.other.theta - s.member.theta) < 0.6, 1, 0),
        reduce='normalized-mean',
        mode=mode,
        include_self=True,
    )
    close = np.abs(theta[None, :] - theta[:, None]) < 0.6
    np.testing.assert_allclose(_drift(system, backend=backend), (close * theta).sum(1) / close.sum(1), atol=1e-12)


@BACKENDS
def test_coupling_circular_mean_gives_the_mean_direction_of_the_other_members(backend):
    """Test that the circular mean over the group gives the direction of the sum of the unit vectors of the others."""
    theta = np.array([0.1, 1.0, 2.5, -2.0])
    system = _group(theta, function=lambda s: s.other.theta, mode='group', reduce='circular-mean')
    sines, cosines = np.sin(theta).sum() - np.sin(theta), np.cos(theta).sum() - np.cos(theta)
    np.testing.assert_allclose(_drift(system, backend=backend), np.arctan2(sines, cosines), atol=1e-12)


def test_coupling_map_form_combines_with_parameters_and_time():
    """Test that the combination of the map form reads the parameters of the interaction and the time."""
    u = np.array([0.0, 1.0, 2.5, 4.0])
    cells = LatticeElement('cells', element=CellElement, space=LatticeSpace(4), values={'u': u})
    coupling = CouplingInteraction(
        'coupling',
        target='cells',
        read='u',
        write='u',
        lag=0.5,
        form='map',
        function=lambda s: s.other.u,
        reduce='mean',
        combine=lambda s, reduction: (1 - s.lag) * s.member.u + s.lag * reduction + s.t,
    )
    system = System(elements=[cells], interactions=[coupling], time=DiscreteTime())
    levels = Simulation(system, dt=1).check().prepare().run(1).trajectory_.values('cells.u')[-1]
    np.testing.assert_allclose(levels, 0.5 * u + 0.25 * (np.roll(u, 1) + np.roll(u, -1)))


@BACKENDS
def test_coupling_to_the_next_member_reads_the_member_after_it(backend):
    """Test that the next mode couples every member to the next member in group order, the first after the last."""
    theta = np.array([0.0, 1.0, 2.5, 4.0])
    system = _group(theta, function=_sine, mode='next')
    np.testing.assert_allclose(_drift(system, backend=backend), np.sin(np.roll(theta, -1) - theta), atol=1e-12)


@BACKENDS
def test_coupling_weights_edges_by_a_state_of_the_edge_group(backend):
    """Test that the weights of an edge state multiply the term of both ends of every edge."""
    graph = nx.gnp_random_graph(12, 0.4, seed=4)
    space = NetworkSpace(graph)
    rng = np.random.default_rng(1)
    theta, weights = rng.uniform(0, 2 * np.pi, 12), rng.uniform(0.5, 1.5, graph.number_of_edges())
    system = System(
        elements=[
            LatticeElement('oscillators', element=OscillatorElement, space=space, values={'theta': theta}),
            GroupElement('links', element=LinkElement, space=space, on='edges', values={'w': weights}),
        ],
        interactions=[WeightedCouplingInteraction('coupling', target='oscillators', links='links')],
    )
    matrix = np.zeros((12, 12))
    sources, targets = space.check().prepare().edges_[None]
    matrix[sources, targets] = matrix[targets, sources] = weights
    expected = (matrix * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)
    np.testing.assert_allclose(_drift(system, backend=backend), expected, atol=1e-12)


def test_coupling_with_a_phase_lag_locks_at_the_lagged_frequency():
    """Test that identical oscillators with a phase lag lock at the frequency `omega - K sin(lag)`."""
    theta = np.random.default_rng(3).uniform(-0.3, 0.3, 10)
    system = _oscillators(
        NetworkSpace(nx.complete_graph(10)),
        theta,
        np.ones(10),
        function=_sine,
        lag=0.5,
        mode='group',
        reduce='mean',
        include_self=True,
    )
    phases = _phases(system, 40.0)
    assert np.ptp(phases[-1]) < 1e-6
    assert phases[-1, 0] - phases[-2, 0] == pytest.approx(1 - math.sin(0.5), abs=1e-6)


def test_coupling_with_a_delay_locks_two_oscillators_at_the_delayed_frequency():
    """Test that two delayed oscillators lock in phase at the frequency `Omega` that solves `Omega = 1 - sin(Omega)`."""
    system = _oscillators(NetworkSpace(nx.path_graph(2)), np.array([0.0, 0.5]), np.ones(2), function=_sine, delay=1.0)
    phases = _phases(system, 60.0)
    frequency = scipy.optimize.brentq(lambda value: value - 1 + math.sin(value), 0.0, 1.0)
    assert abs(phases[-1, 1] - phases[-1, 0]) < 1e-4
    assert phases[-1, 0] - phases[-2, 0] == pytest.approx(frequency, abs=1e-4)


def test_coupling_map_form_combines_the_state_with_the_reduction():
    """Test that the map form sets every level to `combine(u_i, reduction)` on the bonds of a lattice."""
    u = np.array([0.0, 1.0, 2.5, 4.0])
    cells = LatticeElement('cells', element=CellElement, space=LatticeSpace(4), values={'u': u})
    coupling = CouplingInteraction(
        'coupling',
        target='cells',
        read='u',
        write='u',
        reduce='mean',
        include_self=True,
        form='map',
        combine=lambda s, reduction: s.member.u + reduction,
    )
    system = System(elements=[cells], interactions=[coupling], time=DiscreteTime())
    levels = Simulation(system, dt=1).check().prepare().run(1).trajectory_.values('cells.u')[-1]
    np.testing.assert_allclose(levels, (np.roll(u, 1) + np.roll(u, -1) + u) / 3)


def _heading(member):
    """Return the unit heading of a member in the complex plane."""
    return exp(I * member.theta)


def _polarization(s):
    """Return the modulus of the mean heading, the polar order parameter."""
    return abs(s.mean_heading)


def test_coupling_map_form_aligns_a_vicsek_flock_at_low_noise():
    """Test that members taking the mean heading within a unit radius at low noise reach a polar order above 0.9."""
    rng = np.random.default_rng(0)
    values = {'x': rng.uniform(0.0, 10.0, (300, 2)), 'theta': rng.uniform(-math.pi, math.pi, 300), 'v0': 0.5}
    noise = 0.1
    alignment = CouplingInteraction(
        'alignment',
        target='flock',
        read='theta',
        write='theta',
        function=lambda s: vector(SR, [cos(s.other.theta), sin(s.other.theta)]),
        neighborhood=RadiusNeighborhood(1.0),
        include_self=True,
        form='map',
        combine=lambda s, total: atan2(total[1], total[0]) + uniform('turn', -noise * pi, noise * pi),
    )
    system = System(
        elements=[GroupElement('flock', element=MoverElement, size=300, values=values)],
        interactions=[alignment],
        space=ContinuousSpace(extent=(10.0, 10.0), boundary='periodic'),
        time=DiscreteTime(),
    )
    observables = [
        ObservableSpec('mean_heading', _heading, group='flock'),
        ObservableSpec('polarization', _polarization),
    ]
    simulation = Simulation(system, dt=1.0, random_state=0, observables=observables).check().prepare()
    polarization = np.asarray(simulation.run(until=150.0).trajectory_.values('polarization'), dtype=float)
    assert polarization[0] < 0.2
    assert polarization[-1] > 0.9


def test_coupling_over_edges_needs_edges_for_its_rates():
    """Test that a rate per edge over members without edges raises an error."""
    with pytest.raises(ValidationError, match='laws per edge'):
        compile_system(_group(np.zeros(3)))


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'mode': 'chain'}, 'mode of coupling'),
        ({'reduce': 'median'}, 'reduce of coupling'),
        ({'source': 'others'}, 'needs the mode group'),
        ({'mode': 'next', 'reduce': 'max'}, 'sum or mean'),
        ({'mode': 'group', 'reduce': 'max'}, 'include_self'),
        ({'mode': 'group', 'reduce': 'min', 'include_self': True, 'weights': np.ones((2, 2))}, 'no matrix'),
        ({'form': 'jump'}, 'form of coupling'),
        ({'function': 1.0}, 'callable'),
        ({'combine': 1.0}, 'callable'),
        ({'mode': 'group', 'neighborhood': RadiusNeighborhood(1.0)}, 'neighbourhood'),
        ({'mode': 'group', 'weights': 'w'}, 'state of the edges'),
        ({'weights': np.ones((2, 2))}, 'mode group'),
        ({'size': 3}, 'needs the mode hyperedges'),
        ({'mode': 'hyperedges', 'size': 1}, 'whole number of 2 or more'),
        ({'mode': 'hyperedges', 'include_self': True}, 'no include_self'),
        ({'mode': 'hyperedges', 'delay': 1.0}, 'no delay'),
    ],
)
def test_check_rejects_unknown_variants_and_mismatched_structures(arguments, match):
    """Test that unknown variants, functions that cannot be called, and weights that do not suit the mode fail."""
    with pytest.raises(ValidationError, match=match):
        CouplingInteraction('coupling', target='oscillators', **arguments).check()


def test_coupling_separate_elements_to_each_other():
    """Test that a coupling of separate elements gives each the sum of the terms of the other elements."""
    phases = [0.0, 1.0, 2.5]
    oscillators = [OscillatorElement(f'o{index}', theta=phase) for index, phase in enumerate(phases)]
    sine = CouplingInteraction(
        'sine',
        target=['o0', 'o1', 'o2'],
        read='theta',
        write='theta',
        function=lambda s: sin(s.other.theta - s.member.theta),
        mode='group',
    )
    system = System(elements=oscillators, interactions=[sine])
    trajectory = Simulation(system, dt=0.01).check().prepare().run(until=1.0).trajectory_
    found = [trajectory.values(f'o{index}.theta')[-1] for index in range(3)]
    expected = scipy.integrate.solve_ivp(
        lambda _, theta: np.sin(theta[None, :] - theta[:, None]).sum(axis=1),
        (0.0, 1.0),
        phases,
        rtol=1e-10,
        atol=1e-12,
    ).y[:, -1]
    np.testing.assert_allclose(found, expected, atol=1e-6)


def test_coupling_hub_by_largest_value_of_group():
    """Test that a hub element coupled to a group by the largest value gets the largest term over the group."""
    cells = GroupElement('cells', element=CellElement, size=3, values={'u': [1.0, 3.0, 2.0]})
    drive = CouplingInteraction(
        'drive',
        target='hub',
        source='cells',
        read='u',
        write='u',
        function=lambda s: s.other.u,
        reduce='max',
        mode='group',
    )
    system = System(elements=[cells, CellElement('hub')], interactions=[drive])
    trajectory = Simulation(system, dt=0.1).check().prepare().run(until=1.0).trajectory_
    assert trajectory.values('hub.u')[-1] == pytest.approx(3.0)


def test_coupling_group_to_hub_element():
    """Test that the members of a group coupled to a hub element each get the term of the hub."""
    cells = GroupElement('cells', element=CellElement, size=2, values={'u': [0.0, 1.0]})
    inhibition = CouplingInteraction(
        'inhibition',
        target='cells',
        source='hub',
        read='u',
        write='u',
        function=lambda s: -s.other.u,
        mode='group',
    )
    system = System(elements=[cells, CellElement('hub', u=2.0)], interactions=[inhibition])
    trajectory = Simulation(system, dt=0.1).check().prepare().run(until=1.0).trajectory_
    np.testing.assert_allclose(trajectory.values('cells.u')[-1], [-2.0, -1.0])


def test_check_separate_elements_need_group_mode():
    """Test that a coupling of separate elements outside the mode group is rejected."""
    with pytest.raises(ValidationError, match='separate elements'):
        CouplingInteraction('sine', target=['a', 'b'], mode='next').check()


class HeadingElement(Element):
    """Agent with a velocity of two components and no rules of its own."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        vx: Annotated[object, StateSpec()] = 0.0,
        vy: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.vx = vx
        self.vy = vy


def _velocities(locations, velocities, **arguments):
    """Return the velocities of agents after one step of an agent coupling over radii."""
    agents = GroupElement(
        'agents',
        element=HeadingElement,
        size=len(locations),
        values={'vx': [item[0] for item in velocities], 'vy': [item[1] for item in velocities]},
        location=locations,
    )
    coupling = CouplingInteraction('coupling', target='agents', write=('vx', 'vy'), form='agent', **arguments)
    space = ContinuousSpace(extent=(10.0, 10.0))
    system = System(elements=[agents], interactions=[coupling], space=space, time=DiscreteTime())
    snapshot = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_.agents('agents', 1)
    return np.stack([snapshot.values['vx'], snapshot.values['vy']], axis=1)


def test_coupling_agents_over_radii_by_priority():
    """Test that only the radii of the highest priority that hold another agent add their reduced terms."""
    radii = (
        RadiusSpec(1.0, lambda agent, other, model: (1.0, 0.0), priority=1),
        RadiusSpec(5.0, lambda agent, other, model: (0.0, 1.0)),
        RadiusSpec(5.0, lambda agent, other, model: (0.0, 2.0), reduce='mean'),
    )
    found = _velocities([(1.0, 1.0), (1.5, 1.0), (4.0, 1.0)], [(0.0, 0.0)] * 3, radii=radii)
    np.testing.assert_allclose(found, [[1.0, 0.0], [1.0, 0.0], [0.0, 4.0]])


def test_coupling_agents_with_clamped_turn_and_norm():
    """Test that the agent coupling turns the vector by at most the largest angle and clips its norm."""
    radii = (RadiusSpec(5.0, lambda agent, other, model: (-1.0, 3.0)),)
    clamp = ClampSpec(max_angle=0.5, max_norm=2.0)
    found = _velocities([(1.0, 1.0), (2.0, 1.0)], [(1.0, 0.0)] * 2, radii=radii, clamp=clamp)
    np.testing.assert_allclose(found, [[2 * math.cos(0.5), 2 * math.sin(0.5)]] * 2)


def test_coupling_agents_combine_reduction_and_priority():
    """Test that the combination of the agent form reads the reduction and its priority, or `None` for no agent."""
    radii = (RadiusSpec(1.0, lambda agent, other, model: (1.0, 0.0), priority=2),)

    def combine(agent, model, reduction, priority):
        return (0.0, -1.0) if reduction is None else (float(priority), 0.0)

    found = _velocities([(1.0, 1.0), (1.5, 1.0), (8.0, 8.0)], [(0.0, 0.0)] * 3, radii=radii, combine=combine)
    np.testing.assert_allclose(found, [[2.0, 0.0], [2.0, 0.0], [0.0, -1.0]])


@pytest.mark.parametrize(
    ('arguments', 'match'),
    [
        ({'form': 'agent'}, 'at least one radius'),
        ({'clamp': ClampSpec(max_norm=1.0)}, 'agent form only'),
    ],
)
def test_check_agent_form_needs_radii_and_takes_them_alone(arguments, match):
    """Test that the agent form needs radii, and that the other forms take no radii, clamp, or phase."""
    with pytest.raises(ValidationError, match=match):
        CouplingInteraction('coupling', target='agents', **arguments).check()


HYPEREDGES = [(0, 1, 2), (1, 2, 3, 4), (3, 4, 5), (0, 2, 4, 5), (2, 5)]


def _hyperedge_drift(coupling, theta, backend, space=None):
    """Return the drift of oscillators without natural frequencies coupled over the hyperedges of a network."""
    space = space or NetworkSpace(nx.complete_graph(6), hyperedges=HYPEREDGES)
    oscillators = LatticeElement('oscillators', element=OscillatorElement, space=space, values={'theta': theta})
    problem = compile_system(System(elements=[oscillators], interactions=[coupling]), backend=backend)
    return problem.states.read(
        problem.drift(0.0, problem.initial_state, problem.initial_parameters),
        'oscillators.theta',
    )


@BACKENDS
def test_coupling_over_hyperedges_is_the_three_body_kuramoto_drift(backend):
    """Test that the three-body phase coupling over the triangles equals a direct loop over the triangles."""
    theta = np.random.default_rng(5).uniform(0.0, 2 * np.pi, size=6)
    coupling = CouplingInteraction(
        'triads',
        target='oscillators',
        read='theta',
        write='theta',
        mode='hyperedges',
        size=3,
        function=lambda s: sin(s.others[0].theta + s.others[1].theta - 2 * s.member.theta),
    )
    expected = np.zeros(6)
    for hyperedge in (item for item in HYPEREDGES if len(item) == 3):
        for member in hyperedge:
            first, second = (theta[other] for other in hyperedge if other != member)
            expected[member] += np.sin(first + second - 2 * theta[member])
    np.testing.assert_allclose(_hyperedge_drift(coupling, theta, backend), expected, rtol=0.0, atol=1e-12)


@BACKENDS
def test_coupling_over_hyperedges_of_every_size_averages_the_differences(backend):
    """Test that the default term, weighted and averaged over the hyperedges of every size, matches a loop."""
    theta = np.random.default_rng(6).uniform(0.0, 2 * np.pi, size=6)
    coupling = CouplingInteraction(
        'pull',
        target='oscillators',
        read='theta',
        write='theta',
        mode='hyperedges',
        reduce='mean',
        weights=lambda s: count(over=s.members),
        lag=0.2,
    )
    expected, numbers = np.zeros(6), np.zeros(6)
    for hyperedge in HYPEREDGES:
        for member in hyperedge:
            others = [theta[other] for other in hyperedge if other != member]
            expected[member] += len(hyperedge) * sum(other - theta[member] - 0.2 for other in others)
            numbers[member] += 1
    expected = expected / numbers
    np.testing.assert_allclose(_hyperedge_drift(coupling, theta, backend), expected, rtol=0.0, atol=1e-12)


def test_coupling_over_hyperedges_needs_a_network_with_hyperedges():
    """Test that the coupling over hyperedges of members on a lattice raises an error naming the hyperedges."""
    coupling = CouplingInteraction('pull', target='oscillators', read='theta', write='theta', mode='hyperedges')
    with pytest.raises(ValidationError, match='network with hyperedges'):
        _hyperedge_drift(coupling, np.zeros(4), NumpyBackend(), LatticeSpace(4))
