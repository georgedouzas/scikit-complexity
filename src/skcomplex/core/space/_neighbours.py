"""List the pairs of neighbouring members of lattices and networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ..base import ValidationError
from ._cayley import GroupSpace
from ._lattices import LatticeSpace
from ._spaces import NetworkSpace, Space


def neighbour_pairs(space: Space | None, size: int) -> list[tuple[int, int]]:
    """Return the pairs of neighbouring members of a lattice, a group space, or a network, each pair once.

    Args:
        space:
            The lattice or the network.

        size:
            The number of members, one per site or node.

    Returns:
        pairs:
            The member indices of every pair of neighbours.

    Raises:
        ValidationError:
            The space is neither a lattice nor a network.

    Examples:
        >>> from skcomplex.core.space import LatticeSpace, neighbour_pairs
        >>> neighbour_pairs(LatticeSpace(3, boundary='fixed'), 3)
        [(0, 1), (1, 2)]
    """
    if isinstance(space, NetworkSpace):
        index = {node: number for number, node in enumerate(space.graph.nodes)}
        return [(index[first], index[second]) for first, second in space.graph.edges if first != second]
    if isinstance(space, GroupSpace):
        found_edges = {
            frozenset((first, second)): (first, second) for first, second, _ in space.edges if first != second
        }
        return list(found_edges.values())
    if not isinstance(space, LatticeSpace):
        error_msg = f'Neighbours are defined on a LatticeSpace, a GroupSpace, or a NetworkSpace, not on {space!r}.'
        raise ValidationError(error_msg)
    dimensions = space.dimensions
    sites = np.arange(size).reshape(dimensions)
    found: dict[frozenset[int], tuple[int, int]] = {}
    for axis, length in enumerate(dimensions):
        shifted = np.roll(sites, -1, axis=axis)
        pairs = np.stack([sites.ravel(), shifted.ravel()], axis=1)
        if space.boundary != 'periodic':
            edge = np.indices(dimensions)[axis].ravel() == length - 1
            pairs = pairs[~edge]
        for first, second in pairs.tolist():
            if first != second:
                found.setdefault(frozenset((first, second)), (first, second))
    return list(found.values())
