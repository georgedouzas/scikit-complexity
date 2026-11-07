"""Test the `_supply_chains` module of `skcomplex.systems.social.organizations`."""

import networkx as nx
import numpy as np

from skcomplex.systems.social.organizations import SupplyChainDiffusionSystem

from ..conftest import _run, _values


def _network(edges, size):
    """Return a supply network of nodes `0` to `size - 1` with weighted edges from supplier to buyer."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(size))
    graph.add_weighted_edges_from(edges)
    return graph


def test_buyer_forces_suppliers_beholden_to_it():
    """Test that a buyer forces the suppliers whose outflow is below its inflow, and not a resisting one."""
    graph = _network([(1, 0, 2), (2, 0, 6), (1, 3, 4), (2, 3, 1), (4, 1, 1)], 5)
    trajectory = _run(SupplyChainDiffusionSystem(graph, seeds=1), 3)
    np.testing.assert_array_equal(_values(trajectory, 'firms', 'adopted', 1), [1.0, 1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(_values(trajectory, 'firms', 'adopted'), [1.0, 1.0, 1.0, 0.0, 0.0])


def test_buyer_with_one_supplier_never_forces_it():
    """Test that diffusion needs two or more suppliers, since a single supplier sends the whole inflow."""
    trajectory = _run(SupplyChainDiffusionSystem(_network([(1, 0, 5)], 2), seeds=1), 3)
    np.testing.assert_array_equal(_values(trajectory, 'firms', 'adopted'), [1.0, 0.0])


def test_good_conditions_spread_from_the_seed_group():
    """Test that adoption only grows from the seeds, one tier of suppliers per period, until it stops."""
    trajectory = _run(SupplyChainDiffusionSystem(), 10)
    adopters = [_values(trajectory, 'firms', 'adopted', frame).sum() for frame in range(11)]
    assert adopters[0] == 2.0
    assert np.all(np.diff(adopters) >= 0)
    assert adopters[-1] > adopters[1] > adopters[0]
    assert adopters[-1] == adopters[-2]
