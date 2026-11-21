"""Define the neighbourhood of the grid offsets a local rule reads, in order."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from ...errors import ValidationError
from ...types import IntTuple
from ._base import BaseNeighborhood


class LatticeNeighborhood(BaseNeighborhood):
    """A neighbourhood of the ordered lattice offsets that a local rule reads.

    Each offset steps from a site to one neighbour, such as `(-1,)` for the cell on the left, and the offset of zeros
    is the site itself. A local rule receives the values in the order of the offsets, so
    `LatticeNeighborhood([(-1,), (0,), (1,)])` hands it a cell between its left and right neighbours. A cellular
    automaton reads a cell and its two neighbours, a river carries a pollutant from the cells upstream, and a queue of
    traffic reads the car ahead.

    Read more in the user guide, under Spaces.

    Args:
        offsets:
            The offsets, one tuple of integers per neighbour, in the order rules receive their values.

        label:
            The name of the neighbourhood, such as `'moore'`.

    Attributes:
        offsets_:
            The offsets as tuples of integers.

    See Also:
        RadiusNeighborhood: Builds its offsets from a radius and a metric.
        LatticeSpace: Holds the sites the offsets step between.

    Examples:
        >>> from skcomplex.core.components.spaces import LatticeNeighborhood
        >>> LatticeNeighborhood([(-1,), (0,), (1,)]).prepare().offsets_
        ((-1,), (0,), (1,))
    """

    def __init__(self, offsets: tuple[IntTuple, ...], *, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.offsets = offsets

    def _get_offsets(self, n_dim: int | None = None) -> tuple[IntTuple, ...]:
        """Return the offsets as tuples of integers.

        They fix their own number of dimensions.
        """
        return self.prepare().offsets_

    def _join_slots(self, base: Hashable, slot: Hashable) -> IntTuple:
        """Return the sum of an offset and the base offset it is read after.

        Args:
            base:
                The offset read first, such as `(1, 0)`.

            slot:
                The offset read after it, such as `(0, -1)`.

        Returns:
            offset:
                The sum of both, such as `(1, -1)`.
        """
        return tuple(int(step) for step in np.array(base, ndmin=1) + np.array(slot, ndmin=1))

    def _get_from_slots(self, slots: Sequence[Hashable]) -> LatticeNeighborhood:
        """Return the lattice neighbourhood of other offsets."""
        return LatticeNeighborhood(tuple(tuple(int(step) for step in np.array(slot, ndmin=1)) for slot in slots))

    def _prepare(self, context: None = None) -> None:
        """Set `offsets_`, the offsets as tuples of integers.

        It first raises an error for offsets of different numbers of dimensions.
        """
        super()._prepare(context)
        offsets = tuple(tuple(int(step) for step in np.atleast_1d(offset)) for offset in self.offsets)
        if len({len(offset) for offset in offsets}) > 1:
            error_msg = 'Every offset of a neighbourhood has the same number of dimensions.'
            raise ValidationError(error_msg)
        self.offsets_ = offsets
