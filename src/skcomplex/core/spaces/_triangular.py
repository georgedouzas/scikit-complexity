"""Define triangular neighbourhoods, the six neighbours of a site of a triangular lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ...types import IntTuple
from ._base import BaseNeighborhood
from ._lattice_neighborhood import LatticeNeighborhood

TRIANGULAR_OFFSETS = ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))


class TriangularNeighborhood(LatticeNeighborhood):
    """Six neighbours of a site of a triangular lattice, counterclockwise from the direction of the second axis.

    Direction `k` points at `60 k` degrees in the plane, so direction `k + 3` modulo 6 is opposite to direction `k`.

    Args:
        center:
            Whether the site itself comes first.

        label:
            The label of the neighbourhood.

    Examples:
        >>> from skcomplex.core.spaces import TriangularNeighborhood
        >>> TriangularNeighborhood().prepare().offsets_
        ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))
    """

    def __init__(self, *, center: bool = False, label: str = 'neighborhood') -> None:
        BaseNeighborhood.__init__(self, label)
        self.center = center

    def _get_offsets(self) -> tuple[IntTuple, ...]:
        """Return the six offsets, after the site itself when included."""
        return ((0, 0), *TRIANGULAR_OFFSETS) if self.center else TRIANGULAR_OFFSETS
