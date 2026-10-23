"""Test interactions over neighbours that run once per edge and read and change the states of edge groups."""

from typing import ClassVar

import networkx as nx
import numpy as np
import pytest
from sage.all__sagemath_symbolics import sin

from skcomplex.core.base import RoleError, ValidationError, parameter, role
from skcomplex.core.compile import compile_system, validate
from skcomplex.core.components import Element, Group
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Jump, Rate, Update, neighbors
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4
from skcomplex.core.space import DiscreteTime, LatticeSpace, Layer, NetworkSpace
from skcomplex.core.system import System

BACKENDS = ('numpy', 'numba')

Neuron = Element.define('Neuron', states={'a': 0.0})
Synapse = Element.define('Synapse', states={'w': 0.1})
Person = Element.define('Person', states={'infected': 0.0, 'wage': 0.0})
Oscillator = Element.define('Oscillator', states={'theta': 0.0}, parameters={'omega': 0.0})


class Hebbian(Interaction):
    """Weights that grow with the product of the activities at both ends of an edge."""

    over: ClassVar[str] = 'neighbours'
    nodes: str = role()
    synapses: str = role()
    eta: float = parameter(1.0)

    def laws(self, s):
        """Return the growth of the weight of one edge."""
        return [Rate(s.edge.w, s.eta * s.source.a * s.target.a)]


class Infection(Interaction):
    """Contacts recorded on edges, with an update of an end that edges cannot set."""

    over: ClassVar[str] = 'neighbours'
    people: str = role()
    contacts: str = role()

    def laws(self, s):
        """Return the record of a contact and the infection of the source."""
        return [Update(s.edge.contacted, 1.0), Update(s.source.infected, s.target.infected)]


class Contagion(Interaction):
    """Infection along the edges of a layer, without an edge group."""

    over: ClassVar[str] = 'neighbours'
    people: str = role()

    def laws(self, s):
        """Return the infection of both ends by the other."""
        return [
            Rate(s.source.infected, s.target.infected * (1 - s.source.infected)),
            Rate(s.target.infected, s.source.infected * (1 - s.target.infected)),
        ]


class Wages(Interaction):
    """Wages that flow from the source of a work edge to its target."""

    over: ClassVar[str] = 'neighbours'
    people: str = role()
    jobs: str = role()

    def laws(self, s):
        """Return the transfer of wages along one work edge."""
        return [Rate(s.target.wage, s.edge.salary), Rate(s.source.wage, -s.edge.salary)]


class Coupling(Interaction):
    """Kuramoto coupling along the weighted edges of a network."""

    over: ClassVar[str] = 'neighbours'
    oscillators: str = role()
    links: str = role()

    def laws(self, s):
        """Return the pull of both ends towards each other."""
        return [
            Rate(s.source.theta, s.edge.w * sin(s.target.theta - s.source.theta)),
            Rate(s.target.theta, s.edge.w * sin(s.source.theta - s.target.theta)),
        ]


def _hebbian(space, activities):
    """Return a network of neurons with fixed activities and Hebbian synapses on its edges."""
    return System(
        components=[
            Neuron.lattice('neurons', space, a=activities),
            Group('synapses', element=Synapse, space=space, on='edges'),
        ],
        interactions=[Hebbian('hebb', nodes='neurons', synapses='synapses', eta=2.0)],
    )


@pytest.mark.parametrize('backend', BACKENDS)
def test_hebbian_weights_grow_between_coactive_nodes(backend):
    """Test that Hebbian weights end higher on edges between co-active nodes than on edges between inactive ones."""
    space = NetworkSpace(nx.cycle_graph(6))
    activities = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    simulation = Simulation(_hebbian(space, activities), backend=backend, solver=RK4(dt=0.01), dt=0.1)
    weights = simulation.run(until=1.0).trajectory_['synapses.w'][-1]
    sources, targets = space.list_edges()
    coactive = activities[sources] * activities[targets] > 0
    assert np.allclose(weights[coactive], 2.1, atol=1e-9)
    assert np.allclose(weights[~coactive], 0.1, atol=1e-12)
    assert weights[coactive].min() > weights[~coactive].max()


@pytest.mark.parametrize('backend', BACKENDS)
def test_edge_laws_read_the_ends_in_the_direction_of_directed_edges(backend):
    """Test that the source and target of a directed edge follow its direction, and edge states weigh the flow."""
    Job = Element.define('Job', states={'salary': 1.0})
    graph = nx.DiGraph([(0, 1), (2, 1), (1, 3)])
    space = NetworkSpace(graph)
    system = System(
        components=[
            Person.lattice('people', space),
            Group('jobs', element=Job, space=space, on='edges', values={'salary': [1.0, 2.0, 4.0]}),
        ],
        interactions=[Wages('wages', people='people', jobs='jobs')],
    )
    problem = compile_system(system, backend=backend)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    salaries = dict(zip(graph.edges, [1.0, 2.0, 4.0], strict=True))
    expected = np.zeros(4)
    for (first, second), salary in salaries.items():
        expected[first] -= salary
        expected[second] += salary
    assert np.allclose(problem.states.read(drift, 'people.wage'), expected, atol=1e-12)


def test_two_layers_carry_infection_and_wages_apart():
    """Test that infection spreads only along contact edges and wages flow only along work edges."""
    contact = nx.path_graph(5)
    work = nx.Graph([(0, 4), (2, 3)])
    work.add_nodes_from(range(5))
    Job = Element.define('Job', states={'salary': 1.0})
    space = NetworkSpace(contact, layers=[Layer('work', work)])
    system = System(
        components=[
            Person.lattice('people', space, infected=[1.0, 0.0, 0.0, 0.0, 0.0]),
            Group('jobs', element=Job, space=space, on='edges', layer='work'),
        ],
        interactions=[Contagion('contagion', people='people'), Wages('wages', people='people', jobs='jobs')],
    )
    trajectory = Simulation(system, solver=RK4(dt=0.01), dt=0.5).run(until=20.0).trajectory_
    infected, wages = trajectory['people.infected'][-1], trajectory['people.wage'][-1]
    assert np.all(infected > 0.9)
    assert np.allclose(wages, [-20.0, 0.0, -20.0, 20.0, 20.0], atol=1e-9)
    isolated = System(
        components=[
            Person.lattice('people', space, infected=[1.0, 0.0, 0.0, 0.0, 0.0]),
            Group('jobs', element=Job, space=space, on='edges', layer='work'),
        ],
        interactions=[Contagion('contagion', people='people', layer='work')],
    )
    trajectory = Simulation(isolated, solver=RK4(dt=0.01), dt=0.5).run(until=20.0).trajectory_
    assert np.allclose(trajectory['people.infected'][-1], [1.0, 0.0, 0.0, 0.0, 1.0], atol=1e-6)


@pytest.mark.parametrize('backend', BACKENDS)
def test_edge_states_weigh_neighbourhood_reductions(backend):
    """Test that a reduction over a network weighs every neighbour with the state of the edge that joins them."""
    space = NetworkSpace(nx.path_graph(4))
    Weighted = Interaction.define(
        'Weighted',
        roles=['nodes', 'links'],
        laws=lambda s: [Rate(s.nodes.theta, neighbors(s.nodes.other.theta, weight=s.links.w))],
    )
    system = System(
        components=[
            Oscillator.lattice('nodes', space, theta=[1.0, 2.0, 3.0, 4.0]),
            Group('links', element=Synapse, space=space, on='edges', values={'w': [10.0, 100.0, 1000.0]}),
        ],
        interactions=[Weighted('weighted', nodes='nodes', links='links')],
    )
    problem = compile_system(system, backend=backend)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    expected = [10.0 * 2, 10.0 * 1 + 100.0 * 3, 100.0 * 2 + 1000.0 * 4, 1000.0 * 3]
    assert np.allclose(problem.states.read(drift, 'nodes.theta'), expected, atol=1e-9)


@pytest.mark.parametrize('backend', BACKENDS)
def test_weighted_kuramoto_on_edges_matches_the_adjacency_coupling(backend):
    """Test that the coupling written once per edge matches the sum over the weighted adjacency of every node."""
    graph = nx.gnp_random_graph(12, 0.4, seed=4)
    space = NetworkSpace(graph)
    rng = np.random.default_rng(1)
    theta, weights = rng.uniform(0, 2 * np.pi, 12), rng.uniform(0.5, 1.5, graph.number_of_edges())
    system = System(
        components=[
            Oscillator.lattice('oscillators', space, theta=theta),
            Group('links', element=Synapse, space=space, on='edges', values={'w': weights}),
        ],
        interactions=[Coupling('coupling', oscillators='oscillators', links='links')],
    )
    problem = compile_system(system, backend=backend)
    rates = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    drift = problem.states.read(rates, 'oscillators.theta')
    matrix = np.zeros((12, 12))
    sources, targets = space.list_edges()
    matrix[sources, targets] = matrix[targets, sources] = weights
    expected = (matrix * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)
    assert np.allclose(drift, expected, atol=1e-12)


def test_edge_jumps_change_the_ends_of_their_edge():
    """Test that a jump law of an edge changes the states at its ends and that its channels follow the edges."""
    SIS = Element.define('SIS', states={'s': 1.0, 'i': 0.0})
    Spread = Interaction.define(
        'Spread',
        roles=['nodes'],
        parameters={'beta': 1.0},
        over='neighbours',
        laws=lambda s: [Jump(s.beta * s.source.i * s.target.s, {s.target.s: -1, s.target.i: 1})],
    )
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    space = NetworkSpace(graph)
    system = System(
        components=[SIS.lattice('nodes', space, s=[0.0, 1.0, 1.0, 1.0], i=[1.0, 0.0, 0.0, 0.0])],
        interactions=[Spread('spread', nodes='nodes')],
    )
    trajectory = Simulation(system, dt=1.0, random_state=0).run(until=60.0).trajectory_
    assert trajectory['nodes.i'][-1].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_edge_interactions_run_on_lattice_bonds():
    """Test that an interaction over neighbours on a lattice runs once per bond of neighbouring sites."""
    Diffusion = Interaction.define(
        'Diffusion',
        roles=['cells'],
        over='neighbours',
        laws=lambda s: [
            Rate(s.source.theta, s.target.theta - s.source.theta),
            Rate(s.target.theta, s.source.theta - s.target.theta),
        ],
    )
    system = System(
        components=[Oscillator.lattice('cells', LatticeSpace(4), theta=[1.0, 0.0, 0.0, 0.0])],
        interactions=[Diffusion('diffusion', cells='cells')],
    )
    problem = compile_system(system)
    drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
    assert drift.tolist() == [-2.0, 1.0, 0.0, 1.0]


def test_update_laws_set_the_edge_member_only():
    """Test that an update law of an edge sets the edge member, and one that sets an end raises an error."""
    Contact = Element.define('Contact', states={'contacted': 0.0})
    space = NetworkSpace(nx.path_graph(3))
    system = System(
        components=[
            Person.lattice('people', space, infected=[1.0, 0.0, 0.0]),
            Group('contacts', element=Contact, space=space, on='edges'),
        ],
        interactions=[Infection('infection', people='people', contacts='contacts')],
        time=DiscreteTime(),
    )
    with pytest.raises(ValidationError, match='at an end of every edge'):
        compile_system(system)


def test_edge_group_of_another_layer_is_rejected():
    """Test that an interaction that names another layer than its edge group raises a role error."""
    Job = Element.define('Job', states={'salary': 1.0})
    space = NetworkSpace(nx.path_graph(3), layers=[Layer('work', nx.complete_graph(3))])
    system = System(
        components=[Person.lattice('people', space), Group('jobs', element=Job, space=space, on='edges')],
        interactions=[Wages('wages', people='people', jobs='jobs', layer='work')],
    )
    with pytest.raises(RoleError, match="layer 'work'"):
        compile_system(system)


def test_validation_reports_a_missing_layer():
    """Test that an interaction that names a layer the network lacks gets a binding issue listing the layers."""
    space = NetworkSpace(nx.path_graph(3), layers=[Layer('work', nx.complete_graph(3))])
    system = System(
        components=[Person.lattice('people', space)],
        interactions=[Contagion('contagion', people='people', layer='trade')],
    )
    issues = validate(system).issues
    assert [issue.code for issue in issues] == ['binding']
    assert "'work'" in issues[0].message
