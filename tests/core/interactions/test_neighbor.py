"""Test the `_neighbor` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.elements import Element, GroupElement, LatticeElement
from skcomplex.core.interactions import NeighborInteraction
from skcomplex.core.laws import JumpLaw, RateLaw, UpdateLaw
from skcomplex.core.spaces import DiscreteTime, LatticeSpace, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.compile import compile_system
from skcomplex.engine.runtime import Simulation
from skcomplex.engine.solvers import RK4Solver
from skcomplex.errors import RoleError, ValidationError
from skcomplex.specs import LayerSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])


class NeuronElement(Element):
    """Neuron with an activity."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a


class SynapseElement(Element):
    """Synapse with a weight."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        w: Annotated[object, StateSpec()] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.w = w


class PersonElement(Element):
    """Person who may be infected and earns a wage."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        infected: Annotated[object, StateSpec()] = 0.0,
        wage: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.infected = infected
        self.wage = wage


class JobElement(Element):
    """Job with a salary."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        salary: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.salary = salary


class ContactElement(Element):
    """Contact that records whether it happened."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        contacted: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.contacted = contacted


class OscillatorElement(Element):
    """Oscillator with a phase."""

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


class SISElement(Element):
    """Node that is susceptible or infected."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        s: Annotated[object, StateSpec()] = 1.0,
        i: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.s = s
        self.i = i


class HebbianInteraction(NeighborInteraction):
    """Weights that grow with the product of the activities at both ends of an edge."""

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        synapses: Annotated[str, RoleSpec()],
        eta: Annotated[float, ParameterSpec()] = 1.0,
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.nodes = nodes
        self.synapses = synapses
        self.eta = eta

    def _get_laws(self, s):
        """Return the growth of the weight of one edge."""
        return [RateLaw(s.edge.w, s.eta * s.source.a * s.target.a)]


class InfectionInteraction(NeighborInteraction):
    """Contacts recorded on edges, with an update of an end that edges cannot set."""

    def __init__(
        self,
        label: str,
        *,
        people: Annotated[str, RoleSpec()],
        contacts: Annotated[str, RoleSpec()],
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.people = people
        self.contacts = contacts

    def _get_laws(self, s):
        """Return the record of a contact and the infection of the source."""
        return [UpdateLaw(s.edge.contacted, 1.0), UpdateLaw(s.source.infected, s.target.infected)]


class ContagionInteraction(NeighborInteraction):
    """Infection along the edges of a layer, without an edge group."""

    def __init__(
        self,
        label: str,
        *,
        people: Annotated[str, RoleSpec()],
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.people = people

    def _get_laws(self, s):
        """Return the infection of both ends by the other."""
        return [
            RateLaw(s.source.infected, s.target.infected * (1 - s.source.infected)),
            RateLaw(s.target.infected, s.source.infected * (1 - s.target.infected)),
        ]


class WagesInteraction(NeighborInteraction):
    """Wages that flow from the source of a work edge to its target."""

    def __init__(
        self,
        label: str,
        *,
        people: Annotated[str, RoleSpec()],
        jobs: Annotated[str, RoleSpec()],
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.people = people
        self.jobs = jobs

    def _get_laws(self, s):
        """Return the transfer of wages along one work edge."""
        return [RateLaw(s.target.wage, s.edge.salary), RateLaw(s.source.wage, -s.edge.salary)]


class CouplingInteraction(NeighborInteraction):
    """Kuramoto coupling along the weighted edges of a network."""

    def __init__(
        self,
        label: str,
        *,
        oscillators: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.oscillators = oscillators
        self.links = links

    def _get_laws(self, s):
        """Return the pull of both ends towards each other."""
        return [
            RateLaw(s.source.theta, s.edge.w * sin(s.target.theta - s.source.theta)),
            RateLaw(s.target.theta, s.edge.w * sin(s.source.theta - s.target.theta)),
        ]


class SpreadInteraction(NeighborInteraction):
    """Infection of the target of an edge by its source, one jump at a time."""

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        beta: Annotated[float, ParameterSpec()] = 1.0,
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.nodes = nodes
        self.beta = beta

    def _get_laws(self, s):
        """Return the infection jump of the target."""
        return [JumpLaw(s.beta * s.source.i * s.target.s, {s.target.s: -1, s.target.i: 1})]


class DiffusionInteraction(NeighborInteraction):
    """Diffusion between the ends of every bond."""

    def __init__(
        self,
        label: str,
        *,
        cells: Annotated[str, RoleSpec()],
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.cells = cells

    def _get_laws(self, s):
        """Return the flow between both ends."""
        return [
            RateLaw(s.source.theta, s.target.theta - s.source.theta),
            RateLaw(s.target.theta, s.source.theta - s.target.theta),
        ]


def _hebbian(space, activities):
    """Return a network of neurons with fixed activities and Hebbian synapses on its edges."""
    return System(
        elements=[
            LatticeElement('neurons', element=NeuronElement, space=space, values={'a': activities}),
            GroupElement('synapses', element=SynapseElement, space=space, on='edges'),
        ],
        interactions=[HebbianInteraction('hebb', nodes='neurons', synapses='synapses', eta=2.0)],
    )


def test_neighbor_interaction_stores_its_layer():
    """Test that a neighbour interaction keeps the layer it runs on, and none by default."""
    assert ContagionInteraction('contagion', people='people', layer='work').layer == 'work'
    assert NeighborInteraction('empty').check().prepare().run(SimpleNamespace()).laws_ == []


@BACKENDS
def test_hebbian_weights_grow_between_coactive_nodes(backend):
    """Test that Hebbian weights end higher on edges between co-active nodes than on edges between inactive ones."""
    space = NetworkSpace(nx.cycle_graph(6))
    activities = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    simulation = (
        Simulation(_hebbian(space, activities), backend=backend, solver=RK4Solver(dt=0.01), dt=0.1).check().prepare()
    )
    weights = simulation.run(until=1.0).trajectory_.values('synapses.w')[-1]
    sources, targets = space.check().prepare().edges_[None]
    coactive = activities[sources] * activities[targets] > 0
    assert np.allclose(weights[coactive], 2.1, atol=1e-9)
    assert np.allclose(weights[~coactive], 0.1, atol=1e-12)


@BACKENDS
def test_edge_laws_read_the_ends_in_the_direction_of_directed_edges(backend):
    """Test that the source and target of a directed edge follow its direction, and edge states weigh the flow."""
    graph = nx.DiGraph([(0, 1), (2, 1), (1, 3)])
    space = NetworkSpace(graph)
    system = System(
        elements=[
            LatticeElement('people', element=PersonElement, space=space),
            GroupElement('jobs', element=JobElement, space=space, on='edges', values={'salary': [1.0, 2.0, 4.0]}),
        ],
        interactions=[WagesInteraction('wages', people='people', jobs='jobs')],
    )
    problem = compile_system(system, backend=backend)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = np.zeros(4)
    for (first, second), salary in zip(graph.edges, [1.0, 2.0, 4.0], strict=True):
        expected[first] -= salary
        expected[second] += salary
    assert np.allclose(problem.states.read(drift, 'people.wage'), expected, atol=1e-12)


def test_two_layers_carry_infection_and_wages_apart():
    """Test that infection spreads only along contact edges and wages flow only along work edges."""
    work = nx.Graph([(0, 4), (2, 3)])
    work.add_nodes_from(range(5))
    space = NetworkSpace(nx.path_graph(5), layers=[LayerSpec('work', work)])

    def people():
        """Return the people, the first of them infected."""
        return LatticeElement('people', element=PersonElement, space=space, values={'infected': [1.0, 0, 0, 0, 0]})

    jobs = GroupElement('jobs', element=JobElement, space=space, on='edges', layer='work')
    system = System(
        elements=[people(), jobs],
        interactions=[
            ContagionInteraction('contagion', people='people'),
            WagesInteraction('wages', people='people', jobs='jobs'),
        ],
    )
    trajectory = Simulation(system, solver=RK4Solver(dt=0.01), dt=0.5).check().prepare().run(until=20.0).trajectory_
    infected, wages = trajectory.values('people.infected')[-1], trajectory.values('people.wage')[-1]
    assert np.all(infected > 0.9)
    assert np.allclose(wages, [-20.0, 0.0, -20.0, 20.0, 20.0], atol=1e-9)
    isolated = System(
        elements=[people(), jobs],
        interactions=[ContagionInteraction('contagion', people='people', layer='work')],
    )
    trajectory = Simulation(isolated, solver=RK4Solver(dt=0.01), dt=0.5).check().prepare().run(until=20.0).trajectory_
    assert np.allclose(trajectory.values('people.infected')[-1], [1.0, 0.0, 0.0, 0.0, 1.0], atol=1e-6)


@BACKENDS
def test_weighted_kuramoto_on_edges_matches_the_adjacency_coupling(backend):
    """Test that the coupling written once per edge matches the sum over the weighted adjacency of every node."""
    graph = nx.gnp_random_graph(12, 0.4, seed=4)
    space = NetworkSpace(graph)
    rng = np.random.default_rng(1)
    theta, weights = rng.uniform(0, 2 * np.pi, 12), rng.uniform(0.5, 1.5, graph.number_of_edges())
    system = System(
        elements=[
            LatticeElement('oscillators', element=OscillatorElement, space=space, values={'theta': theta}),
            GroupElement('links', element=SynapseElement, space=space, on='edges', values={'w': weights}),
        ],
        interactions=[CouplingInteraction('coupling', oscillators='oscillators', links='links')],
    )
    problem = compile_system(system, backend=backend)
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    drift = problem.states.read(rates, 'oscillators.theta')
    matrix = np.zeros((12, 12))
    sources, targets = space.check().prepare().edges_[None]
    matrix[sources, targets] = matrix[targets, sources] = weights
    expected = (matrix * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)
    assert np.allclose(drift, expected, atol=1e-12)


def test_edge_jumps_change_the_ends_of_their_edge():
    """Test that a jump law of an edge changes the states at its ends and that its channels follow the edges."""
    space = NetworkSpace(nx.DiGraph([(0, 1), (1, 2), (2, 3)]))
    values = {'s': [0.0, 1.0, 1.0, 1.0], 'i': [1.0, 0.0, 0.0, 0.0]}
    system = System(
        elements=[LatticeElement('nodes', element=SISElement, space=space, values=values)],
        interactions=[SpreadInteraction('spread', nodes='nodes')],
    )
    trajectory = Simulation(system, dt=1.0, random_state=0).check().prepare().run(until=60.0).trajectory_
    assert trajectory.values('nodes.i')[-1].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_edge_interactions_run_on_lattice_bonds():
    """Test that an interaction over neighbours on a lattice runs once per bond of neighbouring sites."""
    cells = LatticeElement('cells', element=OscillatorElement, space=LatticeSpace(4), values={'theta': [1.0, 0, 0, 0]})
    problem = compile_system(System(elements=[cells], interactions=[DiffusionInteraction('diffusion', cells='cells')]))
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [-2.0, 1.0, 0.0, 1.0]


def test_update_laws_set_the_edge_member_only():
    """Test that an update law that sets an end of an edge raises an error."""
    space = NetworkSpace(nx.path_graph(3))
    system = System(
        elements=[
            LatticeElement('people', element=PersonElement, space=space, values={'infected': [1.0, 0.0, 0.0]}),
            GroupElement('contacts', element=ContactElement, space=space, on='edges'),
        ],
        interactions=[InfectionInteraction('infection', people='people', contacts='contacts')],
        time=DiscreteTime(),
    )
    with pytest.raises(ValidationError, match='at an end of every edge'):
        compile_system(system)


def test_edge_group_of_another_layer_is_rejected():
    """Test that an interaction that names another layer than its edge group raises a role error."""
    space = NetworkSpace(nx.path_graph(3), layers=[LayerSpec('work', nx.complete_graph(3))])
    system = System(
        elements=[
            LatticeElement('people', element=PersonElement, space=space),
            GroupElement('jobs', element=JobElement, space=space, on='edges'),
        ],
        interactions=[WagesInteraction('wages', people='people', jobs='jobs', layer='work')],
    )
    with pytest.raises(RoleError, match="layer 'work'"):
        compile_system(system)
