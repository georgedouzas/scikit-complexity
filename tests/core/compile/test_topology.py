"""Test variable groups on networks with capacity layouts and the topology requests of update and jump laws."""

from typing import ClassVar

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import as_graph, fit_power_law
from skcomplex.core.base import CapacityError, ComponentKind, SpaceKind, parameter, role
from skcomplex.core.compile import compile_system, missing_declarations, validate
from skcomplex.core.components import Element, Group
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import (
    Jump,
    TopologyRequest,
    Update,
    add_edge,
    add_node,
    neighbors,
    remove_edge,
    remove_node,
    rewire,
    total,
)
from skcomplex.core.runtime import Simulation
from skcomplex.core.solvers import RK4, Gillespie, Iteration
from skcomplex.core.space import Adjacency, DiscreteTime, Layer, NetworkSpace, Radius
from skcomplex.core.system import System

BACKENDS = ('numpy', 'numba')

Node = Element.define('Node', states={'age': 0.0, 'x': 1.0}, laws=lambda s: [Update(s.age, s.age + 1)])
Link = Element.define('Link', states={'w': 1.0})
Growth = Element.define('Growth', laws=lambda s: [Update(add_node(edges=2, toward=lambda node: neighbors(1)), 1)])
SIS = Element.define('SIS', states={'i': 0.0}, parameters={'beta': 0.3, 'gamma': 1.0})


class Rewiring(Interaction):
    """Susceptible ends of edges to infected nodes move them to susceptible nodes at a rate."""

    over: ClassVar[str] = 'neighbours'
    nodes: str = role()
    links: str = role()
    w: float = parameter(0.0)

    def laws(self, s):
        """Return the rewiring of both orientations of a discordant edge."""
        healthy = lambda node: 1 - node.i  # noqa: E731
        return [
            Jump(s.w * (1 - s.source.i) * s.target.i, {rewire(keep='source', toward=healthy): 1}),
            Jump(s.w * s.source.i * (1 - s.target.i), {rewire(keep='target', toward=healthy): 1}),
        ]


def _growing(size, capacity, *, edges=2):
    """Return a network that starts complete on `size` nodes and grows by preferential attachment."""
    space = NetworkSpace.complete(size)
    return System(
        components=[
            Group('nodes', element=Node, space=space, on='sites', variable=True, capacity=capacity),
            Group('links', element=Link, space=space, on='edges', variable=True, capacity=edges * capacity + size**2),
            Growth('growth'),
        ],
        time=DiscreteTime(),
    )


def test_preferential_attachment_gives_a_tail_exponent_near_three():
    """Test that Barabási-Albert growth to 10,000 nodes with two edges each gives a degree tail exponent near 3."""
    simulation = Simulation(_growing(3, 10_003), dt=1000.0, random_state=3).run(until=10_000)
    snapshot = simulation.trajectory_.network(-1)
    degrees = np.bincount(np.concatenate([snapshot.sources, snapshot.targets]))
    assert snapshot.nodes.size == 10_003
    fit = fit_power_law(degrees[degrees > 0], xmin=10, discrete=True, bootstrap=0)
    assert abs(fit.alpha - 3) < 0.3


def test_growth_runs_on_numba_and_agrees_with_numpy():
    """Test that a network growing from 10 to 1,000 nodes runs on Numba with its present nodes, as on NumPy."""
    runs = {
        backend: Simulation(_growing(10, 1000), backend=backend, dt=99.0, random_state=5).run(until=990)
        for backend in BACKENDS
    }
    numpy, numba = (runs[backend].trajectory_ for backend in BACKENDS)
    assert np.allclose(numpy.parameters, numba.parameters, atol=1e-12, equal_nan=True)
    ages = numba['nodes.age']
    for frame, row in enumerate(ages):
        present = numba.network(frame).nodes
        assert np.all(np.isfinite(row[present]))
        assert np.all(np.isnan(np.delete(row, present)))
    assert numba.network(-1).nodes.size == 1000
    assert np.allclose(numpy['nodes.age'], ages, equal_nan=True)


def test_snapshot_by_frame_holds_the_network_of_that_frame():
    """Test that the snapshot of a frame of a growing network holds the nodes and edges present at that frame."""
    trajectory = Simulation(_growing(3, 40), dt=1.0, random_state=0).run(until=30).trajectory_
    snapshot = trajectory.network(10)
    assert snapshot.nodes.tolist() == list(range(13))
    assert snapshot.sources.size == 3 + 2 * 10
    graph = as_graph(snapshot)
    assert nx.is_connected(graph)
    assert graph.number_of_nodes() == 13


def test_a_full_capacity_raises_a_capacity_error():
    """Test that a request beyond the capacity of a variable group names the group and the fix."""
    with pytest.raises(CapacityError, match='capacity'):
        Simulation(_growing(3, 5), dt=1.0, random_state=0).run(until=5)


def test_requests_remove_and_add_edges_and_nodes():
    """Test that update laws of nodes and edges remove edges and nodes, and add edges to present nodes only."""
    space = NetworkSpace(nx.path_graph(6))
    Pruning = Interaction.define(
        'Pruning',
        roles=['nodes', 'links'],
        over='neighbours',
        laws=lambda s: [Update(remove_edge(), s.source.x * s.target.x)],
    )
    Leaving = Element.define('Leaving', states={'age': 0.0, 'x': 0.0}, laws=lambda s: [Update(remove_node(), s.x)])
    system = System(
        components=[
            Group('nodes', element=Leaving, space=space, on='sites', variable=True, values={'x': [1, 1, 0, 0, 0, 1]}),
            Group('links', element=Link, space=space, on='edges', variable=True),
        ],
        interactions=[Pruning('pruning', nodes='nodes', links='links')],
        time=DiscreteTime(),
    )
    trajectory = Simulation(system, dt=1.0, random_state=0).run(until=1).trajectory_
    snapshot = trajectory.network(1)
    assert snapshot.nodes.tolist() == [2, 3, 4]
    assert sorted(zip(snapshot.sources.tolist(), snapshot.targets.tolist(), strict=True)) == [(2, 3), (3, 4)]
    Joining = Element.define('Joining', states={'x': 0.0}, laws=lambda s: [Update(add_edge(w=5.0), s.x)])
    space = NetworkSpace(nx.empty_graph(4))
    system = System(
        components=[
            Group('nodes', element=Joining, space=space, on='sites', variable=True, values={'x': [1, 0, 0, 0]}),
            Group('links', element=Link, space=space, on='edges', variable=True, capacity=10),
        ],
        time=DiscreteTime(),
    )
    trajectory = Simulation(system, dt=1.0, random_state=0).run(until=3).trajectory_
    snapshot = trajectory.network(-1)
    assert np.all(snapshot.sources == 0)
    assert sorted(snapshot.targets.tolist()) == [1, 2, 3]
    assert np.allclose(trajectory['links.w'][-1][snapshot.edges], 5.0)


def _prevalence(w, seed):
    """Return the mean prevalence of SIS with rewiring at rate `w` late in a run on a random graph."""
    space = NetworkSpace.erdos_renyi(200, 0.04, seed=seed)
    infected = np.random.default_rng(seed).random(200) < 0.2
    Spread = Interaction.define(
        'Spread',
        roles=['nodes'],
        laws=lambda s: [
            Jump(s.nodes.beta * (1 - s.nodes.i) * neighbors(s.nodes.other.i), {s.nodes.i: 1}),
            Jump(s.nodes.gamma * s.nodes.i, {s.nodes.i: -1}),
        ],
    )
    system = System(
        components=[
            Group('nodes', element=SIS, space=space, on='sites', variable=True, values={'i': infected.astype(float)}),
            Group('links', element=Link, space=space, on='edges', variable=True),
        ],
        interactions=[Spread('spread', nodes='nodes'), Rewiring('rewiring', nodes='nodes', links='links', w=w)],
    )
    trajectory = Simulation(system, dt=1.0, random_state=seed).run(until=30).trajectory_
    return float(np.nanmean(trajectory['nodes.i'][15:]))


def test_rewiring_raises_the_epidemic_threshold():
    """Test that susceptible nodes rewiring away from infected neighbours suppress an epidemic above its threshold."""
    static, adaptive = _prevalence(0.0, 1), _prevalence(5.0, 1)
    assert static > 0.2
    assert adaptive < 0.5 * static


def test_reductions_over_capacity_groups_count_present_members():
    """Test that a total over a variable group sums the present members only."""
    Counter = Element.define('Counter', states={'size': 0.0})
    Counting = Interaction.define(
        'Counting',
        roles=['counter', 'nodes'],
        laws=lambda s: [Update(s.counter.size, total(s.nodes.x))],
    )
    space = NetworkSpace.complete(3)
    system = System(
        components=[
            Group('nodes', element=Node, space=space, on='sites', variable=True, capacity=10),
            Group('links', element=Link, space=space, on='edges', variable=True, capacity=30),
            Growth('growth'),
            Counter('counter'),
        ],
        interactions=[Counting('counting', counter='counter', nodes='nodes')],
        time=DiscreteTime(),
    )
    trajectory = Simulation(system, dt=1.0, random_state=0).run(until=4).trajectory_
    assert trajectory['counter.size'].tolist() == [0.0, 3.0, 4.0, 5.0, 6.0]
    problem = compile_system(system)
    assert problem.topology is not None


def test_validation_reports_a_request_on_a_fixed_group():
    """Test that a law that requests a new edge on a network with a fixed edge group gets the issue group-fixed."""
    space = NetworkSpace(nx.path_graph(4))
    Joining = Element.define('Joining', states={'x': 1.0}, laws=lambda s: [Update(add_edge(), 0.5)])
    system = System(
        components=[
            Group('nodes', element=Joining, space=space, on='sites', variable=True),
            Group('links', element=Link, space=space, on='edges'),
        ],
        time=DiscreteTime(),
    )
    issues = [issue for issue in validate(system).issues if issue.code == 'group-fixed']
    assert len(issues) == 1
    assert 'links' in issues[0].message
    assert 'variable=True' in issues[0].fix


def test_validation_reports_a_node_request_on_fixed_site_groups():
    """Test that a node request on a network whose site groups are fixed names the fixed groups."""
    space = NetworkSpace(nx.path_graph(4))
    system = System(
        components=[
            Node.lattice('nodes', space),
            Group('links', element=Link, space=space, on='edges', variable=True),
            Growth('growth'),
        ],
        time=DiscreteTime(),
    )
    issues = [issue for issue in validate(system).issues if issue.code == 'group-fixed']
    assert [issue.path for issue in issues] == ['growth']
    assert 'nodes' in issues[0].message


def test_validation_reports_a_metric_neighbourhood_on_a_network():
    """Test that a reduction over a radius on a network gets law-space with a fix that names a layer."""
    Reading = Element.define(
        'Reading',
        states={'x': 0.0},
        laws=lambda s: [Update(s.x, neighbors(s.other.x, over=Radius(1.0)))],
    )
    system = System(components=[Reading.lattice('nodes', NetworkSpace(nx.path_graph(3)))], time=DiscreteTime())
    issues = validate(system).issues
    assert [issue.code for issue in issues] == ['law-space']
    assert 'continuous space' in issues[0].fix
    assert 'layer' in issues[0].fix


def test_validation_reports_a_reduction_over_a_missing_layer():
    """Test that a reduction over a layer the network lacks gets a binding issue that lists its layers."""
    Reading = Element.define(
        'Reading',
        states={'x': 0.0},
        laws=lambda s: [Update(s.x, neighbors(s.other.x, over=Adjacency(layer='trade')))],
    )
    space = NetworkSpace(nx.path_graph(3), layers=[Layer('work', nx.path_graph(3))])
    issues = validate(System(components=[Reading.lattice('nodes', space)], time=DiscreteTime())).issues
    assert [issue.code for issue in issues] == ['binding']
    assert "'work'" in issues[0].message


def test_validation_lists_the_solvers_of_variable_groups():
    """Test that a solver without support for variable groups gets solver-unsupported listing the iteration solver."""
    issues = validate(_growing(3, 10), solver='rk4').issues
    unsupported = [issue for issue in issues if issue.code == 'solver-unsupported']
    assert any('variable-group' in issue.message and 'iteration' in issue.fix for issue in unsupported)


def test_network_primitives_and_solvers_declare_their_requirements_and_support():
    """Test that layers, neighbourhoods, and requests require their spaces, and solvers declare network groups."""
    assert Layer.requires.space == {SpaceKind.NETWORK}
    assert Adjacency.requires.space == {SpaceKind.NETWORK}
    assert Radius.requires.space == {SpaceKind.CONTINUOUS}
    assert TopologyRequest.requires.space == {SpaceKind.NETWORK}
    assert missing_declarations() == []
    for solver in (Iteration, Gillespie):
        assert {ComponentKind.VARIABLE_GROUP, ComponentKind.EDGE_GROUP} <= solver.supports.components
    assert ComponentKind.EDGE_GROUP in RK4.supports.components
    assert ComponentKind.VARIABLE_GROUP not in RK4.supports.components
