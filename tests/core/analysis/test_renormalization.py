"""Test the `_renormalization` module of `skcomplex.core.analysis`."""

from typing import Annotated

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import RenormalizationAnalysis
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import StateSpec, VisualSpec


class AngleElement(Element):
    """Node with an angle."""

    def __init__(self, label: str, visual: VisualSpec | None = None, *, angle: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.angle = angle


def _two_blocks(size: int = 20) -> nx.Graph:
    """Return two complete graphs joined by one edge."""
    graph = nx.disjoint_union(nx.complete_graph(size), nx.complete_graph(size))
    graph.add_edge(0, size)
    return graph


def _ring(size: int) -> nx.Graph:
    """Return a cycle whose nodes hold their position as an angle."""
    graph = nx.cycle_graph(size)
    nx.set_node_attributes(graph, {node: float(node) for node in graph}, 'angle')
    return graph


def test_run_laplacian_merges_two_blocks_into_two_nodes_joined_by_one_edge():
    """Test that two complete graphs of 20 nodes joined by one edge renormalize to two nodes and one edge."""
    analysis = RenormalizationAnalysis(steps=1).prepare(_two_blocks()).run()
    following = analysis.graphs_[1]
    assert following.number_of_nodes() == 2
    assert following.number_of_edges() == 1
    assert nx.number_of_selfloops(following) == 0


def test_run_laplacian_keeps_a_network_without_a_merging_peak():
    """Test that a graph without edges stays as it is at every scale."""
    analysis = RenormalizationAnalysis(steps=2).prepare(nx.empty_graph(5)).run()
    assert [graph.number_of_nodes() for graph in analysis.graphs_] == [5, 5, 5]


def test_run_laplacian_merges_nothing_at_a_short_time():
    """Test that a short diffusion time keeps every node on its own."""
    analysis = RenormalizationAnalysis(steps=1, time=1e-3).prepare(_two_blocks()).run()
    assert analysis.graphs_[1].number_of_nodes() == 40


def test_run_laplacian_merges_at_a_given_time():
    """Test that a time on the plateau between the two scales of the blocks gives the two blocks."""
    analysis = RenormalizationAnalysis(steps=1, time=1.0).prepare(_two_blocks()).run()
    assert analysis.graphs_[1].number_of_nodes() == 2


def test_run_laplacian_acts_on_vectors_above_a_few_thousand_nodes():
    """Test that the sparse exponential of a large cycle at a short time keeps every node on its own."""
    analysis = RenormalizationAnalysis(steps=1, time=1e-2).prepare(nx.cycle_graph(3001)).run()
    assert analysis.graphs_[1].number_of_nodes() == 3001


def test_run_geometric_merges_blocks_of_consecutive_angles():
    """Test that supernodes hold consecutive nodes by angle, link when any of their nodes link, and keep the order."""
    analysis = RenormalizationAnalysis('geometric', steps=3).prepare(_ring(16)).run()
    assert [graph.number_of_nodes() for graph in analysis.graphs_] == [16, 8, 4, 2]
    assert analysis.graphs_[1].number_of_edges() == 8
    assert nx.number_of_selfloops(analysis.graphs_[1]) == 0
    angles = [analysis.graphs_[1].nodes[node]['angle'] for node in range(8)]
    assert angles == pytest.approx([0.5 + 2 * index for index in range(8)])


def test_run_geometric_weighs_the_angle_of_a_supernode_by_the_degrees():
    """Test that a supernode sits at the angles weighted by `k^max(1, beta)`, the plain mean without edges."""
    graph = nx.star_graph(3)
    graph.add_node(4)
    graph.add_node(5)
    nx.set_node_attributes(graph, {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}, 'angle')
    for beta, first in ((1.0, 1 / 4), (2.0, 1 / 10)):
        merged = RenormalizationAnalysis('geometric', steps=1, beta=beta).prepare(graph).run().graphs_[1]
        assert merged.nodes[0]['angle'] == pytest.approx(first)
        assert merged.nodes[1]['angle'] == pytest.approx(2.5)
        assert merged.nodes[2]['angle'] == pytest.approx(4.5)


def test_run_geometric_keeps_the_nodes_left_over_in_the_last_block():
    """Test that the last supernode holds the nodes left over."""
    analysis = RenormalizationAnalysis('geometric', steps=1, block=3).prepare(_ring(7)).run()
    assert analysis.graphs_[1].number_of_nodes() == 3


def test_run_geometric_reads_a_named_angle():
    """Test that the angle attribute is a setting."""
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, {0: 0.0, 1: 3.0, 2: 1.0, 3: 2.0}, 'phase')
    analysis = RenormalizationAnalysis('geometric', steps=1, angle='phase').prepare(graph).run()
    assert analysis.graphs_[1].number_of_edges() == 1


def test_run_geometric_raises_without_angles():
    """Test that a graph without the angle attribute raises an error."""
    with pytest.raises(AnalysisError, match="no attribute 'angle'"):
        RenormalizationAnalysis('geometric').prepare(nx.path_graph(4)).run()


def test_run_measures_every_scale():
    """Test that the degrees, exponents, and clustering hold one entry per scale."""
    analysis = RenormalizationAnalysis('geometric', steps=2).prepare(_ring(64)).run()
    assert [degrees.tolist() for degrees in analysis.degrees_][1] == [2] * 32
    assert analysis.clustering_.tolist() == [0.0, 0.0, 0.0]
    assert np.isnan(analysis.exponents_).all()


def test_prepare_reads_the_last_frame_of_a_trajectory_with_its_node_states():
    """Test that a growing network gives the graph of its last frame with the values of its group on the nodes."""
    nodes = GroupElement('nodes', element=AngleElement, on='sites', variable=True, capacity=20)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=40)
    growth = TopologyInteraction('growth', action='add-node', values={'angle': lambda rng: rng.uniform(0.0, 6.0)})
    space = NetworkSpace(nx.complete_graph(2))
    system = System(elements=[nodes, links], interactions=[growth], space=space, time=DiscreteTime())
    trajectory = Simulation(system, random_state=0).prepare().run(until=6).trajectory_
    analysis = RenormalizationAnalysis('geometric', steps=1).prepare(trajectory).run()
    assert analysis.graphs_[0].number_of_nodes() == 8
    assert analysis.graphs_[1].number_of_nodes() == 4
    assert analysis.graphs_[0].nodes[7]['angle'] > 0


@pytest.mark.parametrize(
    ('kind', 'settings', 'message'),
    [
        ('spectral', {}, 'kind'),
        ('laplacian', {'steps': 0}, 'steps'),
        ('geometric', {'block': 1}, 'block'),
        ('laplacian', {'time': -1.0}, 'time'),
        ('geometric', {'beta': 0.0}, 'beta'),
        ('laplacian', {'beta': 2.0}, 'does not read beta'),
        ('laplacian', {'block': 3}, 'does not read block'),
        ('geometric', {'time': 1.0}, 'does not read time'),
    ],
)
def test_check_rejects_invalid_settings(kind: str, settings: dict[str, float], message: str):
    """Test that an unknown kind, a bad setting, and a setting the kind does not read raise an error."""
    with pytest.raises(AnalysisError, match=message):
        RenormalizationAnalysis(kind, **settings).prepare()
