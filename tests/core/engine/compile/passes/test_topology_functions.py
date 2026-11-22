"""Test variable groups on networks with capacity layouts and the topology requests of update and jump laws."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import fit_law
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import JumpLaw, TopologyRequestSpec, UpdateLaw
from skcomplex.core.components.laws.utils import neighbors, total
from skcomplex.core.components.spaces import (
    AdjacencyNeighborhood,
    LayerSpec,
    NetworkSpace,
    RadiusNeighborhood,
)
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.systems import System
from skcomplex.core.types import Role


def _degree(node):
    """Return the degree of a candidate node, the weight of preferential attachment."""
    return neighbors(1)


def _healthy(node):
    """Return whether a candidate node is susceptible, the weight of a rewired end."""
    return 1 - node.i


def _aging(label='nodes'):
    """Return the aging of every node by one per step."""
    return Interaction('aging', roles={'target': label}, laws=lambda s: [UpdateLaw(s.target.age, s.target.age + 1)])


def _network(space, nodes=None, links=None):
    """Return the nodes with an age and a value, and the links with a weight, of a network space."""
    nodes = Element('nodes', values={'age': 0.0, 'x': 1.0} | (nodes or {}), space=space)
    links = Element('links', values={'w': 1.0} | (links or {}), space=space, on='edges')
    return [nodes, links]


def _growth() -> Interaction:
    """Return the growth that adds a node with two edges toward high degrees at every step."""
    return Interaction(
        'growth',
        roles={'nodes': 'nodes'},
        laws=[UpdateLaw(TopologyRequestSpec('add-node', edges=2, toward=_degree, one=True), 1)],
    )


class RewiringInteraction(Interaction):
    """Susceptible ends of edges to infected nodes move them to susceptible nodes at a rate."""

    def __init__(self, label: str, *, among: Role, along: bool = True, w: float = 0.0) -> None:
        super().__init__(label, among=among, along=along)
        self.w = w

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rewiring of both orientations of a discordant edge."""
        super()._prepare(context)
        s = context
        self.laws_ = [
            JumpLaw(s.w * (1 - s.source.i) * s.target.i, {TopologyRequestSpec('rewire', toward=_healthy): 1}),
            JumpLaw(
                s.w * s.source.i * (1 - s.target.i),
                {TopologyRequestSpec('rewire', keep='target', toward=_healthy): 1},
            ),
        ]


def _growing(size):
    """Return a network that starts complete on `size` nodes and grows by preferential attachment."""
    return System(elements=_network(NetworkSpace(nx.complete_graph(size))), interactions=[_aging(), _growth()])


@pytest.mark.slow
def test_preferential_attachment_gives_a_tail_exponent_near_three():
    """Test that Barabási-Albert growth to 10,000 nodes with two edges each gives a degree tail exponent near 3."""
    trajectory = _growing(3).prepare().simulate(until=10_000, dt=1000.0, random_state=3)
    snapshot = trajectory.get_network(-1)
    degrees = np.bincount(np.concatenate([snapshot.sources, snapshot.targets]))
    assert snapshot.nodes.size == 10_003
    fit = fit_law(degrees[degrees > 0], minimum=10, discrete=True, bootstrap=0)
    assert abs(fit.exponent - 3) < 0.3


def test_growth_runs_on_numba_and_agrees_with_numpy():
    """Test that a network growing from 10 to 1,000 nodes runs on Numba with its present nodes, as on NumPy."""
    numpy, numba = (
        _growing(10).set_params(backend=backend).prepare().simulate(until=990, dt=99.0, random_state=5)
        for backend in (NumpyBackend(), NumbaBackend())
    )
    assert np.allclose(numpy.parameters, numba.parameters, atol=1e-12, equal_nan=True)
    ages = numba.get_values('nodes.age')
    for frame, row in enumerate(ages):
        present = numba.get_network(frame).nodes
        assert np.all(np.isfinite(row[present]))
        assert np.all(np.isnan(np.delete(row, present)))
    assert numba.get_network(-1).nodes.size == 1000
    assert np.allclose(numpy.get_values('nodes.age'), ages, equal_nan=True)


def test_snapshot_by_frame_holds_the_network_of_that_frame():
    """Test that the snapshot of a frame of a growing network holds the nodes and edges present at that frame."""
    trajectory = _growing(3).prepare().simulate(until=30, dt=1.0, random_state=0)
    snapshot = trajectory.get_network(10)
    assert snapshot.nodes.tolist() == list(range(13))
    assert snapshot.sources.size == 3 + 2 * 10
    graph = snapshot.get_graph()
    assert nx.is_connected(graph)
    assert graph.number_of_nodes() == 13


def test_requests_remove_and_add_edges_and_nodes():
    """Test that update laws of nodes and edges remove edges and nodes, and add edges to present nodes only."""
    pruning = Interaction(
        'pruning',
        among='nodes',
        along=True,
        laws=lambda s: [UpdateLaw(TopologyRequestSpec('remove-edge'), s.source.x * s.target.x)],
    )
    leaving = Interaction(
        'leaving',
        roles={'target': 'nodes'},
        laws=lambda s: [UpdateLaw(TopologyRequestSpec('remove-node'), s.target.x)],
    )
    elements = _network(NetworkSpace(nx.path_graph(6)), {'x': np.array([1, 1, 0, 0, 0, 1])})
    system = System(elements=elements, interactions=[_aging(), leaving, pruning])
    trajectory = system.prepare().simulate(until=1, dt=1.0, random_state=0)
    snapshot = trajectory.get_network(1)
    assert snapshot.nodes.tolist() == [2, 3, 4]
    assert sorted(zip(snapshot.sources.tolist(), snapshot.targets.tolist(), strict=True)) == [(2, 3), (3, 4)]

    joining = Interaction(
        'joining',
        roles={'target': 'nodes'},
        laws=lambda s: [UpdateLaw(TopologyRequestSpec('add-edge', values=(('w', 5.0),)), s.target.x)],
    )
    elements = _network(NetworkSpace(nx.empty_graph(4)), {'x': np.array([1, 0, 0, 0])})
    trajectory = System(elements=elements, interactions=[joining]).prepare().simulate(until=3, dt=1.0, random_state=0)
    snapshot = trajectory.get_network(-1)
    assert np.all(snapshot.sources == 0)
    assert sorted(snapshot.targets.tolist()) == [1, 2, 3]
    assert np.allclose(trajectory.get_values('links.w')[-1][snapshot.edges], 5.0)


def _prevalence(w, seed):
    """Return the mean prevalence of SIS with rewiring at rate `w` late in a run on a random graph."""
    space = NetworkSpace(nx.gnp_random_graph(200, 0.04, seed=seed))
    infected = np.random.default_rng(seed).random(200) < 0.2

    def spread(s):
        nodes = s.target
        return [
            JumpLaw(nodes.beta * (1 - nodes.i) * neighbors(nodes.other.i), {nodes.i: 1}),
            JumpLaw(nodes.gamma * nodes.i, {nodes.i: -1}),
        ]

    nodes = Element('nodes', values={'i': infected.astype(float), 'beta': 0.3, 'gamma': 1.0}, space=space)
    links = Element('links', values={'w': 1.0}, space=space, on='edges')
    system = System(
        elements=[nodes, links],
        interactions=[
            Interaction('spread', roles={'target': 'nodes'}, laws=spread),
            RewiringInteraction('rewiring', among='nodes', w=w),
        ],
    )
    trajectory = system.prepare().simulate(until=30, dt=1.0, random_state=seed)
    return float(np.nanmean(trajectory.get_values('nodes.i')[15:]))


def test_rewiring_raises_the_epidemic_threshold():
    """Test that susceptible nodes rewiring away from infected neighbours suppress an epidemic above its threshold."""
    static, adaptive = _prevalence(0.0, 1), _prevalence(5.0, 1)
    assert static > 0.2
    assert adaptive < 0.5 * static


def test_reductions_over_growing_groups_count_present_members():
    """Test that a total over a growing group sums the present members only."""
    counting = Interaction(
        'counting',
        roles={'counter': 'counter', 'nodes': 'nodes'},
        laws=lambda s: [UpdateLaw(s.counter.count, total(s.nodes.x))],
    )
    elements = [*_network(NetworkSpace(nx.complete_graph(3))), Element('counter', values={'count': 0.0})]
    system = System(elements=elements, interactions=[_aging(), _growth(), counting]).prepare()
    trajectory = system.simulate(until=4, dt=1.0, random_state=0)
    assert trajectory.get_values('counter.count').tolist() == [0.0, 3.0, 4.0, 5.0, 6.0]
    assert system._problem.topology is not None


def _reading(over, space):
    """Return nodes whose value is the reduction of the values of their neighbours over a neighbourhood."""
    reading = Interaction(
        'reading',
        roles={'target': 'nodes'},
        laws=lambda s: [UpdateLaw(s.target.x, neighbors(s.target.other.x, over=over))],
    )
    return System(elements=[Element('nodes', values={'x': 0.0}, space=space)], interactions=[reading])


def test_validation_reports_a_metric_neighbourhood_on_a_network():
    """Test that a reduction over a radius on a network gets law-space with a fix that names a layer."""
    system = _reading(RadiusNeighborhood(1.0), NetworkSpace(nx.path_graph(3)))
    issues = PassCompiler().prepare(system).report_.issues
    assert [issue.code for issue in issues] == ['law-space']
    assert 'continuous space' in issues[0].fix
    assert 'layer' in issues[0].fix


def test_validation_reports_a_reduction_over_a_missing_layer():
    """Test that a reduction over a layer the network lacks gets a binding issue that lists its layers."""
    space = NetworkSpace(nx.path_graph(3), layers=[LayerSpec('work', nx.path_graph(3))])
    report = PassCompiler().prepare(_reading(AdjacencyNeighborhood(layer='trade'), space)).report_
    issues = report.issues
    assert [issue.code for issue in issues] == ['binding']
    assert "'work'" in issues[0].message
