"""Define box neighbourhoods, every site within a radius in lexicographic order with the site itself in place."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools

from ...types import IntTuple
from ._base import BaseNeighborhood
from ._lattice_neighborhood import LatticeNeighborhood


class BoxNeighborhood(LatticeNeighborhood):
    """Every site within a radius along every dimension, the site itself in place, as one-dimensional automata read.

    For radius 1 in one dimension the order is left, centre, right, the order of Wolfram's rule numbers.

    Args:
        radius:
            The largest offset along any dimension.

        n_dim:
            The number of dimensions.

        label:
            The label of the neighbourhood.

    Examples:
        >>> from skcomplex.core.spaces import BoxNeighborhood
        >>> BoxNeighborhood().check().prepare().offsets_
        ((-1,), (0,), (1,))
    """

    def __init__(self, radius: int = 1, *, n_dim: int = 1, label: str = 'neighborhood') -> None:
        BaseNeighborhood.__init__(self, label)
        self.radius = radius
        self.n_dim = n_dim

    def _get_offsets(self) -> tuple[IntTuple, ...]:
        """Return every offset within the radius in lexicographic order, the zero offset included."""
        return tuple(itertools.product(range(-self.radius, self.radius + 1), repeat=self.n_dim))
