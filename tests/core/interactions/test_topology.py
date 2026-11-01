"""Test the `_topology` module of `skcomplex.core.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import BaseInteraction, HyperedgeInteraction, TopologyInteraction
from skcomplex.core.laws import UpdateLaw
from skcomplex.core.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import RoleSpec, StateSpec, VisualSpec

BACKENDS = pytest.mark.parametrize('backend', [NumpyBackend(), NumbaBackend()], ids=['numpy', 'numba'])


class MarkedElement(Element):
    """Node with a mark."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        mark: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.mark = mark


class TriangleElement(Element):
    """Triangle with its age in steps."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.age = age


class AgeingInteraction(HyperedgeInteraction):
    """Ageing of every triangle by one per step."""

    def __init__(
        self,
        label: str,
        *,
        nodes: Annotated[str, RoleSpec()],
        triangles: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label, size=3)
        self.nodes = nodes
        self.triangles = triangles

    def _get_laws(self, s):
        """Return the ageing of the triangle."""
        return [UpdateLaw(s.hyperedge.age, s.hyperedge.age + 1)]


class WebInteraction(TopologyInteraction):
    """Topology change that binds its nodes as `web`."""

    def __init__(self, label: str, *, web: Annotated[str, RoleSpec()], **arguments: object) -> None:
        super().__init__(label, target=web, **arguments)
        self.web = web

    def _get_role(self, role: str) -> str:
        """Return `web` for the target."""
        return 'web' if role == 'target' else role


def _triangles(*interactions: BaseInteraction, capacity: int = 60) -> System:
    """Return a network of one triangle whose groups on nodes, edges, and triangles are variable."""
    nodes = GroupElement('nodes', element=MarkedElement, on='sites', variable=True, capacity=capacity)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=3 * capacity)
    triangles = GroupElement('triangles', element=TriangleElement, on='hyperedges', variable=True, capacity=capacity)
    space = NetworkSpace(nx.complete_graph(3), hyperedges=[[0, 1, 2]])
    return System(elements=[nodes, links, triangles], interactions=list(interactions), space=space, time=DiscreteTime())


def _run(graph: nx.Graph, change: TopologyInteraction, until: float, dt: float = 1.0, **values: object) -> Trajectory:
    """Return the trajectory of a network whose variable groups a topology change changes, one frame per step."""
    nodes = GroupElement('nodes', element=MarkedElement, on='sites', variable=True, capacity=500, values=values)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=500)
    system = System(elements=[nodes, links], interactions=[change], space=NetworkSpace(graph))
    return Simulation(system, dt=dt, random_state=0).check().prepare().run(until=until).trajectory_


def _sizes(trajectory: Trajectory, frame: int) -> tuple[int, int]:
    """Return the numbers of nodes and edges of the network at a frame."""
    network = trajectory.network(frame)
    return len(network.nodes), len(network.sources)


def test_get_laws_add_node_runs_once_per_node():
    """Test that a node action with probability one adds one node per node of a path, not one per edge."""
    change = TopologyInteraction('growth', target='nodes', action='add-node', edges=1)
    assert _sizes(_run(nx.path_graph(10), change, 1), 1) == (20, 19)


def test_get_laws_add_node_without_target_adds_one_node_per_step():
    """Test that a node addition without a target adds one node per step from outside the network."""
    change = TopologyInteraction('growth', action='add-node', edges=2)
    assert _sizes(_run(nx.complete_graph(3), change, 5), 5) == (8, 13)


def test_get_laws_draws_the_values_of_new_nodes():
    """Test that a value given as a function of the generator is drawn for every new node."""
    change = TopologyInteraction(
        'growth',
        target='nodes',
        action='add-node',
        values={'mark': lambda rng: rng.uniform(1.0, 2.0)},
    )
    trajectory = _run(nx.empty_graph(20), change, 1)
    marks = np.asarray(trajectory.values('nodes.mark')[-1])[20:40]
    assert len(set(marks.tolist())) == 20
    assert ((marks >= 1.0) & (marks <= 2.0)).all()


def test_get_laws_remove_node_removes_the_nodes_where_the_condition_holds():
    """Test that a conditional removal removes exactly the marked nodes."""
    marks = [1.0 if node % 2 else 0.0 for node in range(10)]
    change = TopologyInteraction('removal', target='nodes', action='remove-node', condition=lambda s: s.source.mark > 0)
    assert _sizes(_run(nx.path_graph(10), change, 1, mark=marks), 1)[0] == 5


def test_get_laws_replace_node_keeps_the_number_of_nodes():
    """Test that replacing every marked node keeps the number of nodes and adds the edges of the newcomers."""
    marks = [1.0] * 3 + [0.0] * 7
    change = TopologyInteraction(
        'turnover',
        target='nodes',
        action='replace-node',
        edges=2,
        condition=lambda s: s.source.mark > 0,
    )
    nodes, _ = _sizes(_run(nx.cycle_graph(10), change, 1, mark=marks), 1)
    assert nodes == 10


def test_get_laws_add_edge_with_a_probability_adds_edges_per_node():
    """Test that nodes adding an edge with a probability add about that share of edges per step."""
    change = TopologyInteraction('linking', target='nodes', action='add-edge', probability=0.5)
    _, edges = _sizes(_run(nx.empty_graph(100), change, 2), 2)
    assert 70 < edges < 130


def test_get_laws_remove_edge_removes_every_edge():
    """Test that removing edges with probability one empties the network in one step."""
    change = TopologyInteraction('expiry', target='nodes', action='remove-edge')
    trajectory = _run(nx.cycle_graph(8), change, 1)
    assert _sizes(trajectory, 0) == (8, 8)
    assert _sizes(trajectory, 1) == (8, 0)


def test_get_laws_rewire_moves_edges_towards_the_weight():
    """Test that rewiring moves the edges between unmarked ends to marked nodes and keeps their number."""
    marks = [1.0] * 5 + [0.0] * 15
    change = TopologyInteraction(
        'rewiring',
        target='nodes',
        action='rewire',
        probability=0.5,
        condition=lambda s: (1 - s.source.mark) * (1 - s.target.mark) > 0,
        toward=lambda node: node.mark,
        keep='source',
    )
    trajectory = _run(nx.cycle_graph(20), change, 5, mark=marks)
    graph = trajectory.network(5).graph()
    assert graph.number_of_edges() == 20
    marked = set(range(5))
    assert all(source in marked or target in marked for source, target in graph.edges)


def test_get_laws_rate_removes_edges_by_jumps():
    """Test that edges removed at a rate decay as `exp(-rate t)`."""
    change = TopologyInteraction('decay', target='nodes', action='remove-edge', rate=0.5)
    trajectory = _run(nx.cycle_graph(400), change, 2.0, dt=0.05)
    assert _sizes(trajectory, 40)[1] / 400 == pytest.approx(np.exp(-1.0), abs=0.06)


@BACKENDS
def test_get_laws_add_node_with_a_hyperedge_attaches_a_triangle_per_step(backend):
    """Test that 50 nodes added with triangles to one triangle give 51 triangles, each new node in one of them."""
    growth = TopologyInteraction('growth', action='add-node', hyperedge=3)
    ageing = AgeingInteraction('ageing', nodes='nodes', triangles='triangles')
    simulation = Simulation(_triangles(growth, ageing), backend=backend, random_state=0).check().prepare()
    trajectory = simulation.run(until=50).trajectory_
    network = trajectory.network(-1)
    graph = network.graph()
    assert network.hyperedges.shape == (51, 3)
    assert (len(network.nodes), len(network.sources)) == (53, 3 + 2 * 50)
    assert set(range(3, 53)) <= set(network.hyperedges.ravel().tolist())
    assert all(graph.has_edge(first, second) for first, second, _ in network.hyperedges)
    assert all(graph.has_edge(member, new) for *ends, new in network.hyperedges for member in ends)
    ages = np.asarray(trajectory.values('triangles.age')[-1])
    assert sorted(ages[~np.isnan(ages)].tolist()) == list(range(51))


def test_get_laws_remove_node_removes_the_hyperedges_that_hold_it():
    """Test that a removed node takes the triangles that hold it with it."""
    removal = TopologyInteraction(
        'removal',
        target='nodes',
        action='remove-node',
        condition=lambda s: s.source.mark > 0,
    )
    system = _triangles(removal)
    system.elements[0].values = {'mark': [1.0, 0.0, 0.0]}
    trajectory = Simulation(system, random_state=0).check().prepare().run(until=1).trajectory_
    assert trajectory.network(0).hyperedges.shape == (1, 3)
    assert trajectory.network(1).hyperedges.shape == (0, 3)


def test_get_laws_add_node_with_a_hyperedge_needs_a_hyperedge_group_of_its_size():
    """Test that growth by hyperedges of another size than the group, or without one, is reported."""
    growth = TopologyInteraction('growth', action='add-node', hyperedge=4)
    with pytest.raises(ValidationError, match='size 3'):
        Simulation(_triangles(growth), random_state=0).check().prepare()
    nodes = GroupElement('nodes', element=MarkedElement, on='sites', variable=True, capacity=8)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=24)
    system = System(elements=[nodes, links], interactions=[growth], space=NetworkSpace(nx.complete_graph(3)))
    with pytest.raises(ValidationError, match='variable group on its hyperedges'):
        Simulation(system, random_state=0).check().prepare()


def test_check_rejects_an_edge_action_without_target():
    """Test that an action other than a node addition needs a target."""
    with pytest.raises(ValidationError):
        TopologyInteraction('change', action='add-edge').check()


@pytest.mark.parametrize(
    'arguments',
    [
        {'action': 'merge'},
        {'rate': 1.0, 'probability': 0.5},
        {'condition': 1},
        {'toward': 'degree'},
        {'keep': 'both'},
        {'edges': -1},
        {'action': 'add-edge', 'hyperedge': 3},
        {'action': 'add-node', 'hyperedge': 1},
    ],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that an unknown action or end, a rate with a probability, bad functions, and bad edges are rejected."""
    with pytest.raises(ValidationError):
        TopologyInteraction('change', target='nodes', **arguments).check()


def test_get_laws_reads_a_renamed_target_end():
    """Test that a rewiring that binds its nodes as `web` reads the target end as `web` and rewires as the plain one."""
    marks = [1.0] * 5 + [0.0] * 15
    arguments = {'action': 'rewire', 'probability': 0.5, 'toward': lambda node: node.mark, 'keep': 'source'}
    plain = TopologyInteraction(
        'rewiring',
        target='nodes',
        condition=lambda s: (1 - s.source.mark) * (1 - s.target.mark) > 0,
        **arguments,
    )
    renamed = WebInteraction(
        'rewiring',
        web='nodes',
        condition=lambda s: (1 - s.source.mark) * (1 - s.web.mark) > 0,
        **arguments,
    )
    graphs = [
        sorted(_run(nx.cycle_graph(20), change, 5, mark=marks).network(5).graph().edges) for change in (plain, renamed)
    ]
    assert graphs[1] == graphs[0]
