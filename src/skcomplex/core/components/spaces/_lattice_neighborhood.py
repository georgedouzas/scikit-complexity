"""Define lattice neighbourhoods.

They are the ordered offsets of the sites a local rule reads.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ...errors import ValidationError
from ...types import IntTuple
from ._base import BaseNeighborhood


class LatticeNeighborhood(BaseNeighborhood):
    """Ordered offsets of the sites a local rule reads.

    The site itself is the zero offset when included.

    Args:
        offsets:
            The offsets, one tuple of integers per neighbour, in the order rules receive their values.

        label:
            The label of the neighbourhood.

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

    def _check(self) -> None:
        """Raise an error for offsets of different numbers of dimensions."""
        if len({len(offset) for offset in self._get_offsets()}) > 1:
            error_msg = 'Every offset of a neighbourhood has the same number of dimensions.'
            raise ValidationError(error_msg)

    def _prepare(self, context: None = None) -> None:
        """Set `offsets_`, the offsets as tuples of integers."""
        self.offsets_ = self._get_offsets()
