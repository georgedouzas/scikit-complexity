"""Test the `_autocatalytic` module of `skcomplex.core.analysis`."""

import networkx as nx
import numpy as np
import pytest

from skcomplex.core.analysis import autocatalytic_sets
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AnalysisError
from skcomplex.core.specs import NetworkSnapshotSpec

REACTIONS = np.array([[0, 1, 2, -1], [2, 0, 3, -1], [3, 1, 2, 4], [5, 6, -1, -1]])


def _snapshot(sources, targets):
    """Return seven nodes, four reactions, and catalysis edges from sources to the products they catalyse."""
    sources, targets = np.asarray(sources, dtype=np.intp), np.asarray(targets, dtype=np.intp)
    return NetworkSnapshotSpec(np.arange(7), sources, targets, np.arange(sources.size), REACTIONS)


def test_largest_raf_holds_the_catalysed_reactions_reachable_from_the_food_set():
    """Test that the largest RAF holds the reactions catalysed within it, and leaves out one without a catalyst."""
    analysis = autocatalytic_sets(_snapshot([3, 4, 0], [2, 3, 4]))
    assert analysis.reactions[0].tolist() == [0, 1, 2]
    assert analysis.products[0].tolist() == [2, 3, 4]
    assert analysis.sizes.tolist() == [3]
    assert analysis.fractions.tolist() == [0.75]
    assert analysis.exists.tolist() == [True]


def test_largest_raf_removes_reactions_whose_catalyst_is_lost():
    """Test that a reaction without a catalyst takes the reactions that need its product with it."""
    analysis = autocatalytic_sets(_snapshot([3, 4], [2, 3]))
    assert analysis.sizes.tolist() == [0]
    assert analysis.exists.tolist() == [False]


def test_food_set_can_be_given():
    """Test that a given food set replaces the nodes that no reaction produces."""
    snapshot = _snapshot([3, 4, 0, 1], [2, 3, 4, 6])
    assert autocatalytic_sets(snapshot).reactions[0].tolist() == [0, 1, 2, 3]
    assert autocatalytic_sets(snapshot, food=[0, 1]).reactions[0].tolist() == [0, 1, 2]


def test_reads_the_frames_of_a_trajectory_and_rejects_other_sources():
    """Test that the analysis reads chosen frames of a growing hypergraph, and rejects a graph."""
    nodes = GroupElement('nodes', element=Element, on='sites', variable=True, capacity=30)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=900)
    groups = GroupElement('functions', element=Element, on='hyperedges', variable=True, size=0, capacity=30, rank=3)
    growth = TopologyInteraction(
        'growth',
        action='add-node',
        hyperedge=3,
        independent=True,
        direction='both',
        toward=lambda node: 0.5,
    )
    space = NetworkSpace(nx.empty_graph(3, create_using=nx.DiGraph))
    system = System(elements=[nodes, links, groups], interactions=[growth], space=space, time=DiscreteTime())
    trajectory = simulate(system, until=20, random_state=0)
    analysis = autocatalytic_sets(trajectory, frames=[0, -1])
    assert analysis.sizes[0] == 0
    assert analysis.sizes[1] > 10
    with pytest.raises(AnalysisError, match='network snapshot'):
        autocatalytic_sets(nx.path_graph(3))
