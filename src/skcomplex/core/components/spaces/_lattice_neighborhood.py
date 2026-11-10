"""Define the neighbourhood of ordered offsets that a local rule reads."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...errors import ValidationError
from ...types import IntTuple
from ._base import BaseNeighborhood


class LatticeNeighborhood(BaseNeighborhood):
    """A neighbourhood of the ordered lattice offsets that a local rule reads.

    Each offset is a vector of integers from a site to a neighbour, and the site itself is the zero offset when it is
    included. The order of the offsets is the order in which a local rule receives the values.

    Unlike `BallNeighborhood`, which builds its offsets from a radius, it takes any list of offsets. Choose it when a
    neighbourhood has an irregular shape, or an order of its own, such as a cell and its two neighbours from left to
    right.

    It appears in elementary cellular automata, which read a cell and its two neighbours, and in rules that read cells
    in one direction only, such as the upwind cells of a model of advection.

    Args:
        offsets:
            The offsets, one tuple of integers per neighbour, in the order rules receive their values.

        label:
            The name of the neighbourhood, such as `'moore'`.

    Attributes:
        offsets_:
            The offsets as tuples of integers.

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeNeighborhood
        >>> LatticeNeighborhood([(0, 1), (1, 0)]).prepare().offsets_
        ((0, 1), (1, 0))
    """

    def __init__(self, offsets: tuple[IntTuple, ...], *, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.offsets = offsets

    def _get_offsets(self) -> tuple[IntTuple, ...]:
        """Return the offsets as tuples of integers."""
        return tuple(tuple(int(step) for step in np.atleast_1d(offset)) for offset in self.offsets)

    def _prepare(self, context: None = None) -> None:
        """Set `offsets_`, the offsets as tuples of integers.

        It first raises an error for offsets of different numbers of dimensions.
        """
        super()._prepare(context)
        if len({len(offset) for offset in self._get_offsets()}) > 1:
            error_msg = 'Every offset of a neighbourhood has the same number of dimensions.'
            raise ValidationError(error_msg)
        self.offsets_ = self._get_offsets()
