"""Test the `_topology` module of `skcomplex.core.interactions`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import TopologyInteraction
from skcomplex.core.spaces import NetworkSpace
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation, Trajectory
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec


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


@pytest.mark.parametrize(
    'arguments',
    [
        {'action': 'merge'},
        {'rate': 1.0, 'probability': 0.5},
        {'condition': 1},
        {'toward': 'degree'},
        {'keep': 'both'},
        {'edges': -1},
    ],
)
def test_check_rejects_invalid_arguments(arguments):
    """Test that an unknown action or end, a rate with a probability, bad functions, and bad edges are rejected."""
    with pytest.raises(ValidationError):
        TopologyInteraction('change', target='nodes', **arguments).check()
