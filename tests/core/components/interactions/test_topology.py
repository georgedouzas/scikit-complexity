"""Test the `_topology` module of `skcomplex.core.components.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import BaseInteraction, HyperedgeInteraction, TopologyInteraction
from skcomplex.core.components.laws import RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import neighbors, where
from skcomplex.core.components.spaces import ContinuousTime, DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.runtime import Simulation, Trajectory
from skcomplex.core.engine.solvers import MesaSolver
from skcomplex.core.errors import RoleError, SolverError, ValidationError
from skcomplex.core.specs import RoleSpec, StateSpec, VisualSpec

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
    return Simulation(system, dt=dt, random_state=0).prepare().run(until=until).trajectory_


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
    simulation = Simulation(_triangles(growth, ageing), backend=backend, random_state=0).prepare()
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
    trajectory = Simulation(system, random_state=0).prepare().run(until=1).trajectory_
    assert trajectory.network(0).hyperedges.shape == (1, 3)
    assert trajectory.network(1).hyperedges.shape == (0, 3)


def test_get_laws_add_node_with_a_hyperedge_needs_a_hyperedge_group_of_its_size():
    """Test that growth by hyperedges of another size than the group, or without one, is reported."""
    growth = TopologyInteraction('growth', action='add-node', hyperedge=4)
    with pytest.raises(ValidationError, match='size 3'):
        Simulation(_triangles(growth), random_state=0).prepare()
    nodes = GroupElement('nodes', element=MarkedElement, on='sites', variable=True, capacity=8)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=24)
    system = System(elements=[nodes, links], interactions=[growth], space=NetworkSpace(nx.complete_graph(3)))
    with pytest.raises(ValidationError, match='variable group on its hyperedges'):
        Simulation(system, random_state=0).prepare()


def test_check_rejects_an_edge_action_without_target():
    """Test that an action other than a node addition needs a target."""
    with pytest.raises(ValidationError):
        TopologyInteraction('change', action='add-edge').prepare()


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
        {'action': 'add-hyperedge'},
        {'action': 'remove-hyperedge', 'hyperedge': 3},
        {'direction': 'sideways'},
    ],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that an unknown action or end, a rate with a probability, bad functions, and bad edges are rejected."""
    with pytest.raises(ValidationError):
        TopologyInteraction('change', target='nodes', **arguments).prepare()


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


class CallableMarkingInteraction(HyperedgeInteraction):
    """Marking of the members of every triangle at a rate a callable gives."""

    def __init__(self, label: str, *, nodes: Annotated[str, RoleSpec()], triangles: Annotated[str, RoleSpec()]) -> None:
        super().__init__(label, size=3)
        self.nodes = nodes
        self.triangles = triangles

    def _get_laws(self, s):
        """Return a rate of the mark of the member whose value is a callable of time and values."""
        return [RateLaw(s.member.mark, lambda t, values: 1.0)]


def test_get_laws_refuses_a_callable_value_on_a_growing_hypergraph():
    """Test that a callable rate of the members of hyperedges that change during a run raises an error."""
    growth = TopologyInteraction('growth', action='add-node', hyperedge=3, rate=1.0)
    marking = CallableMarkingInteraction('marking', nodes='nodes', triangles='triangles')
    system = _triangles(growth, marking)
    system.time = ContinuousTime()
    with pytest.raises(ValidationError, match='switches off only a symbolic value'):
        Simulation(system, random_state=0).prepare()


def _grown(change: TopologyInteraction, seeds: int = 80, capacity: int = 200, **arguments: object) -> Simulation:
    """Return a simulation of an empty network of nodes with evenly spaced marks that a topology change grows."""
    marks = np.linspace(0.0, 2 * np.pi, seeds, endpoint=False)
    nodes = GroupElement('nodes', element=MarkedElement, on='sites', variable=True, capacity=capacity)
    nodes = nodes.set_params(values={'mark': marks.tolist()})
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=3 * capacity)
    space = NetworkSpace(nx.empty_graph(seeds))
    system = System(elements=[nodes, links], interactions=[change], space=space, time=DiscreteTime())
    return Simulation(system, random_state=0, **arguments).prepare()


def _mark_gaps(trajectory: Trajectory) -> np.ndarray:
    """Return the difference of the marks of the ends of every edge of the last frame."""
    network = trajectory.network(-1)
    marks = np.asarray(trajectory.values('nodes.mark')[-1])
    return np.abs(marks[network.sources] - marks[network.targets])


@BACKENDS
def test_get_laws_add_node_attaches_by_the_drawn_values_of_the_new_node(backend):
    """Test that a weight of 1 within an angular distance of 0.1 and 0 elsewhere gives edges within that distance."""
    growth = TopologyInteraction(
        'growth',
        action='add-node',
        values={'mark': lambda rng: rng.uniform(0.0, 2 * np.pi - 0.1)},
        toward=lambda node: where(abs(node.mark - node.source.mark) <= 0.1, 1, 0),
    )
    trajectory = _grown(growth, backend=backend).run(until=100).trajectory_
    gaps = _mark_gaps(trajectory)
    assert gaps.size == 100
    assert (gaps <= 0.1).all()


def test_get_laws_add_node_weights_differ_without_the_source():
    """Test that uniform weights, which read no new node, join marks far apart for the same draws."""
    growth = TopologyInteraction(
        'growth',
        action='add-node',
        values={'mark': lambda rng: rng.uniform(0.0, 2 * np.pi - 0.1)},
    )
    assert (_mark_gaps(_grown(growth).run(until=100).trajectory_) > 0.1).any()


def test_get_laws_add_node_condition_reads_the_drawn_values():
    """Test that a condition on the drawn value adds only the nodes that meet it, with the value it read."""
    growth = TopologyInteraction(
        'growth',
        action='add-node',
        edges=0,
        values={'mark': lambda rng: rng.uniform(0.0, 1.0)},
        condition=lambda s: s.source.mark < 0.5,
    )
    trajectory = _grown(growth, seeds=1).run(until=100).trajectory_
    marks = np.asarray(trajectory.values('nodes.mark')[-1])[1:]
    added = marks[~np.isnan(marks)]
    assert 20 < added.size < 80
    assert (added < 0.5).all()


def test_get_laws_add_node_probability_reads_the_drawn_values():
    """Test that a probability that is the drawn value adds the nodes of high values more often."""
    growth = TopologyInteraction(
        'growth',
        action='add-node',
        edges=0,
        values={'mark': lambda rng: rng.uniform(0.0, 1.0)},
        probability=lambda s: s.source.mark,
    )
    marks = np.asarray(_grown(growth, seeds=1, capacity=400).run(until=300).trajectory_.values('nodes.mark')[-1])[1:]
    added = marks[~np.isnan(marks)]
    assert added.mean() == pytest.approx(2 / 3, abs=0.05)


def test_get_laws_add_node_condition_reads_the_defaults_of_values_not_given():
    """Test that the source of a condition reads the default of a state the action does not draw."""
    growth = TopologyInteraction('growth', action='add-node', edges=0, condition=lambda s: s.source.mark < 0)
    assert _sizes(_grown(growth, seeds=1).run(until=5).trajectory_, -1) == (1, 0)


def test_get_laws_add_node_target_keeps_the_acting_node_as_the_source_of_the_condition():
    """Test that a condition of an addition with a target reads the node it runs for, not the new node."""
    marks = [1.0 if node < 3 else 0.0 for node in range(10)]
    growth = TopologyInteraction(
        'growth',
        target='nodes',
        action='add-node',
        edges=0,
        condition=lambda s: s.source.mark > 0,
    )
    assert _sizes(_run(nx.empty_graph(10), growth, 1, mark=marks), 1) == (13, 0)


def test_get_laws_add_edge_weight_reads_the_adding_node_as_the_source():
    """Test that the weight of a node adding edges reads that node as the source."""
    close = TopologyInteraction(
        'close',
        target='nodes',
        action='add-edge',
        probability=0.5,
        toward=lambda node: where(abs(node.mark - node.source.mark) <= 0.2, 1, 0),
    )
    gaps = _mark_gaps(_grown(close).run(until=1).trajectory_)
    assert gaps.size > 0
    assert (gaps <= 0.2).all()


def test_get_laws_rewire_weight_reads_the_kept_end_as_the_source():
    """Test that the weight of a rewiring reads the end it keeps as the source."""
    rewire = TopologyInteraction(
        'rewire',
        target='nodes',
        action='rewire',
        toward=lambda node: where(abs(node.mark - node.source.mark) <= 0.1, 1, 0),
    )
    graph = nx.empty_graph(4)
    graph.add_edge(0, 2)
    network = _run(graph, rewire, 1, mark=[0.0, 0.05, 5.0, 5.05]).network(-1)
    assert (network.sources.tolist(), network.targets.tolist()) == ([0], [1])


def test_get_laws_replace_node_weight_reads_the_new_node_as_the_source():
    """Test that the condition of a replacement reads the old node and the weight the new one."""
    replacement = TopologyInteraction(
        'replacement',
        target='nodes',
        action='replace-node',
        condition=lambda s: s.source.mark > 9,
        values={'mark': 1.0},
        toward=lambda node: where(abs(node.mark - node.source.mark) <= 0.1, 1, 0),
    )
    marks = [1.0, 1.05, 3.0, 3.05, 10.0]
    network = _run(nx.empty_graph(5), replacement, 1, mark=marks).network(1)
    assert network.sources.size == 1
    assert {2, 3}.isdisjoint(network.sources.tolist() + network.targets.tolist())


def test_get_laws_rate_cannot_read_the_new_node():
    """Test that a rate of an addition that reads the new node raises an error when it is evaluated."""
    growth = TopologyInteraction('growth', action='add-node', edges=0, rate=lambda s: s.source.mark + 1)
    nodes = GroupElement('nodes', element=MarkedElement, on='sites', variable=True, capacity=20)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=20)
    system = System(elements=[nodes, links], interactions=[growth], space=NetworkSpace(nx.empty_graph(2)))
    with pytest.raises(ValidationError, match=r'namespace\.source'):
        Simulation(system, random_state=0).prepare().run(until=2)


def test_get_laws_reduction_cannot_read_the_new_node():
    """Test that a reduction of the weight that reads the new node raises an error when the system compiles."""
    growth = TopologyInteraction(
        'growth',
        action='add-node',
        toward=lambda node: neighbors(node.other.mark + node.source.mark),
    )
    links = nx.path_graph(4)
    nodes = GroupElement('nodes', element=MarkedElement, on='sites', variable=True, capacity=20)
    edges = GroupElement('links', element=Element, on='edges', variable=True, capacity=20)
    system = System(elements=[nodes, edges], interactions=[growth], space=NetworkSpace(links), time=DiscreteTime())
    with pytest.raises(ValidationError, match=r'neighbourhood reduction of nodes reads mark_nodes__source'):
        Simulation(system, random_state=0).prepare()


def test_get_laws_add_node_runs_on_no_mesa_path():
    """Test that the Mesa solver takes no topology change, so no Mesa path reads the new node."""
    growth = TopologyInteraction('growth', action='add-node')
    nodes = GroupElement('nodes', element=MarkedElement, on='sites', variable=True, capacity=20)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=20)
    system = System(elements=[nodes, links], interactions=[growth], space=NetworkSpace(nx.path_graph(4)))
    with pytest.raises(SolverError, match='does not support the laws UpdateLaw'):
        Simulation(system, solver=MesaSolver(), random_state=0).prepare()


def _ranked(*interactions: BaseInteraction, directed: bool = False, capacity: int = 80) -> System:
    """Return five nodes without hyperedges whose variable group on hyperedges holds every size up to four."""
    nodes = GroupElement('nodes', element=MarkedElement, on='sites', variable=True, capacity=capacity)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=capacity * capacity)
    groups = GroupElement('groups', element=Element, on='hyperedges', variable=True, size=0, capacity=capacity, rank=4)
    graph = nx.empty_graph(5, create_using=nx.DiGraph if directed else nx.Graph)
    return System(
        elements=[nodes, links, groups],
        interactions=list(interactions),
        space=NetworkSpace(graph),
        time=DiscreteTime(),
    )


@BACKENDS
def test_get_laws_add_node_forms_hyperedges_of_every_size_up_to_the_rank(backend):
    """Test that new nodes of hyperedges of two, three, and four members fill a ranked group, padded with -1."""
    growth = [TopologyInteraction(f'growth_{size}', action='add-node', hyperedge=size, edges=0) for size in (2, 3, 4)]
    trajectory = Simulation(_ranked(*growth), backend=backend, random_state=0).prepare().run(until=10).trajectory_
    network = trajectory.network(-1)
    sizes = (network.hyperedges >= 0).sum(axis=1)
    assert network.hyperedges.shape == (30, 4)
    assert sorted(np.bincount(sizes).tolist()) == [0, 0, 10, 10, 10]
    assert all(row[size - 1] == max(row[:size]) for row, size in zip(network.hyperedges, sizes, strict=True))
    assert len(network.nodes) == 35
    assert len(network.sources) == 0


def test_get_laws_add_node_joins_independent_ends_both_ways():
    """Test that independent ends of a probability join about that share of the nodes, in each direction apart."""
    growth = TopologyInteraction(
        'growth',
        action='add-node',
        edges=0,
        independent=True,
        direction='both',
        toward=lambda node: 0.5,
    )
    trajectory = Simulation(_ranked(growth, directed=True), random_state=0).prepare().run(until=60).trajectory_
    network = trajectory.network(-1)
    edges = set(zip(network.sources.tolist(), network.targets.tolist(), strict=True))
    pairs = 65 * 64 / 2 - 5 * 4 / 2
    assert len(edges) == len(network.sources)
    assert len(edges) / (2 * pairs) == pytest.approx(0.5, abs=0.05)
    assert len({(first, second) for first, second in edges if (second, first) in edges}) > 0


def test_get_laws_add_hyperedge_and_remove_hyperedge_change_the_hyperedges():
    """Test that nodes form triangles that dissolve, so the number of hyperedges settles near its balance."""
    form = TopologyInteraction('form', target='nodes', action='add-hyperedge', hyperedge=3, probability=0.2)
    dissolve = TopologyInteraction('dissolve', target='triangles', action='remove-hyperedge', probability=0.25)
    system = _triangles(form, dissolve, capacity=200)
    trajectory = Simulation(system, random_state=0).prepare().run(until=200).trajectory_
    counts = [trajectory.network(frame).hyperedges.shape[0] for frame in range(100, 201)]
    assert np.mean(counts) == pytest.approx(3 * 0.2 / 0.25, rel=0.3)
    network = trajectory.network(-1)
    assert len(network.nodes) == 3
    assert len(network.sources) == 3
    assert all(len(set(row.tolist())) == 3 for row in network.hyperedges)


def test_get_laws_hyperedges_beyond_the_rank_or_interactions_on_a_ranked_group_are_reported():
    """Test that a hyperedge larger than the rank and laws per hyperedge on a ranked group are rejected."""
    growth = TopologyInteraction('growth', action='add-node', hyperedge=5, edges=0)
    with pytest.raises(ValidationError, match='rank'):
        Simulation(_ranked(growth), random_state=0).prepare()
    system = _ranked(AgeingInteraction('ageing', nodes='nodes', triangles='groups'))
    system.space = NetworkSpace(nx.complete_graph(5), hyperedges=[[0, 1, 2], [1, 2, 3, 4]])
    system.elements[2].size = None
    with pytest.raises(RoleError, match='rank'):
        Simulation(system, random_state=0).prepare()
