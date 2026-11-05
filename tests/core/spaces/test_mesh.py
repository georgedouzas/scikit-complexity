"""Test the `_mesh` module of `skcomplex.core.spaces`."""

import numpy as np
import skfem

from skcomplex.core.spaces import MeshSpace


def _square():
    """Return the mesh space of a unit square split into eight triangles."""
    grid = skfem.MeshTri.init_tensor(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    return MeshSpace(grid.p, grid.t).prepare()


def test_mesh_space_prepares_its_nodes_and_boundary():
    """Test that a mesh of a unit square has its nodes as sites and every node but the centre on the boundary."""
    space = _square()
    center = int(np.flatnonzero((space.positions_ == 0.5).all(axis=1))[0])
    assert (space.size_, space.n_dim_, space.shape_, space.periodic_) == (9, 2, (9,), False)
    assert sorted(space.boundary_nodes_.tolist()) == [node for node in range(9) if node != center]


def test_mesh_space_neighbours_share_a_cell():
    """Test that the neighbours of a node are the other nodes of the cells that hold it, in increasing order."""
    space = _square()
    center = int(np.flatnonzero((space.positions_ == 0.5).all(axis=1))[0])
    neighbours = space._get_neighbors(center)
    assert center not in neighbours.tolist()
    assert len(neighbours) >= 4
    assert np.array_equal(neighbours, np.sort(neighbours))
