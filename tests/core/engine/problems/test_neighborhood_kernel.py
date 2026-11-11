"""Test the `neighborhood_kernel` module."""

from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_gap import libgap

from skcomplex.core.components.elements import Element
from skcomplex.core.components.spaces import (
    AdjacencyNeighborhood,
    CayleySpace,
    LatticeNeighborhood,
    LatticeSpace,
    NetworkSpace,
    RadiusNeighborhood,
)
from skcomplex.core.engine._utils import declarations_of
from skcomplex.core.engine.problems._neighborhood_kernel import NeighborhoodKernel
from skcomplex.core.enums import VariableKind
from skcomplex.core.specs import StateSpec


def test_neighborhood_kernel_get_neighbor_sets():
    """Test `NeighborhoodKernel._get_neighbor_sets` on a small case."""
    indptr, others = np.array([0, 3, 4]), np.array([1, 2, 3, 0])
    indptr, sets = NeighborhoodKernel._get_neighbor_sets(indptr, others, 2)
    assert (indptr.tolist(), sets.tolist()) == ([0, 3, 3], [[1, 2], [1, 3], [2, 3]])


def test_neighborhood_kernel_get_paths_of():
    """Test `NeighborhoodKernel._get_paths_of` on a small case."""
    readers, others = np.array([0, 1, 2]), np.array([1, 2, 3])
    assert NeighborhoodKernel._get_paths_of('path-betweenness', readers, others, 5).tolist() == [
        0.0,
        2.0,
        2.0,
        0.0,
        0.0,
    ]


def test_neighborhood_kernel_get_components_of():
    """Test `NeighborhoodKernel._get_components_of` on a small case."""
    readers, others = np.array([1, 3, 4]), np.array([2, 4, 5])
    assert NeighborhoodKernel._get_components_of('component-label', readers, others, 6).tolist() == [
        0.0,
        1.0,
        1.0,
        3.0,
        3.0,
        3.0,
    ]
    assert NeighborhoodKernel._get_components_of('component-size', readers, others, 6).tolist() == [
        1.0,
        2.0,
        2.0,
        3.0,
        3.0,
        3.0,
    ]
    assert NeighborhoodKernel._get_components_of('component-giant', readers, others, 6).tolist() == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]


def test_neighborhood_kernel_reduce_pairs():
    """Test `NeighborhoodKernel._reduce_pairs` on a small case."""
    readers, counts = np.array([0, 0, 2]), np.array([2, 0, 1])
    values = np.array([1.0, 3.0, 5.0])
    assert NeighborhoodKernel._reduce_pairs('mean', readers, np.ones(3), values, counts).tolist() == [2.0, 0.0, 5.0]
    assert NeighborhoodKernel._reduce_pairs('argmax', readers, np.array([0.5, 2.0, 1.0]), values, counts).tolist() == [
        3.0,
        0.0,
        5.0,
    ]


def test_neighborhood_kernel_get_network_rows():
    """Test `NeighborhoodKernel._get_network_rows` on a small case."""
    space = NetworkSpace(nx.path_graph(3)).prepare()
    assert [item.tolist() for item in NeighborhoodKernel._get_network_rows(space, AdjacencyNeighborhood())] == [
        [0, 1, 3, 4],
        [1, 0, 2, 1],
        [0, 0, 1, 1],
    ]


def test_neighborhood_kernel_get_lattice_rows():
    """Test `NeighborhoodKernel._get_lattice_rows` on a small case."""
    space = LatticeSpace(3, boundary='fixed').prepare()
    rows = NeighborhoodKernel._get_lattice_rows(space, LatticeNeighborhood(((-1,), (1,))))
    assert [item.tolist() for item in rows] == [[0, 1, 3, 4], [1, 0, 2, 1]]


def test_neighborhood_kernel_get_default_neighborhood():
    """Test `NeighborhoodKernel._get_default_neighborhood` on a small case."""
    lattice = LatticeSpace((4, 4)).prepare()
    assert len(NeighborhoodKernel._get_default_neighborhood(lattice).prepare().offsets_) == 8
    cayley = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=1).prepare()
    assert NeighborhoodKernel._get_default_neighborhood(cayley).elements == ('a', 'a^-1', 'b', 'b^-1')


def test_neighborhood_kernel_get_cone_direction():
    """Test `NeighborhoodKernel._get_cone_direction` on a small case."""

    class BoidElement(Element):
        def __init__(
            self,
            label,
            x: Annotated[object, StateSpec(vector=True, role='position')] = 0.0,
            theta: Annotated[float, StateSpec()] = 0.0,
        ):
            super().__init__(label)
            self.x = x
            self.theta = theta

    states = declarations_of(BoidElement, VariableKind.STATE)
    assert (
        NeighborhoodKernel._get_cone_direction(RadiusNeighborhood(1.0, angle=1.0, direction='theta'), states, 2)
        == 'theta'
    )


def test_neighborhood_kernel_get_spaces_of():
    """Test `NeighborhoodKernel._get_spaces_of` on a small case."""
    assert (
        str(NeighborhoodKernel._get_spaces_of(RadiusNeighborhood(1.0)))
        == "(<class 'skcomplex.core.components.spaces._continuous.ContinuousSpace'>,)"
    )
