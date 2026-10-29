"""Define continuous spaces, boxes of real positions or the whole of R^d."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

import numpy as np
import numpy.typing as npt

from ...errors import ValidationError
from ._base import BaseNeighborhood, BaseSpace

BOUNDARIES = ('open', 'periodic', 'reflecting')


class ContinuousSpace(BaseSpace):
    """Continuous space of real positions, a box with a boundary or the whole of `R^d`.

    A box has an extent along every dimension, from 0 to the extent. Its boundary acts on every position state of the
    members on it: `'open'` leaves positions alone, `'periodic'` joins opposite sides and measures pair distances by the
    minimum image, and `'reflecting'` mirrors positions back into the box and reverses the velocity across the wall.
    Without an extent the space is unbounded and has `n_dim` dimensions. The positions are states of the members, so
    the space has no sites, and a site has no neighbours in it.

    Args:
        extent:
            The side of the box along every dimension, or empty for an unbounded space.

        n_dim:
            The number of dimensions of an unbounded space, or `None` for the number of sides of the box.

        boundary:
            `'open'`, `'periodic'`, or `'reflecting'`.

        label:
            The label of the space in its system.

    Attributes:
        extent_:
            The side of the box along every dimension, or `()` for an unbounded space.

    Examples:
        >>> from skcomplex.core.spaces import ContinuousSpace
        >>> ContinuousSpace(extent=(10.0, 5.0), boundary='periodic').check().prepare().extent_
        (10.0, 5.0)
        >>> ContinuousSpace(n_dim=3).check().prepare().n_dim_
        3
    """

    def __init__(
        self,
        extent: Sequence[float] = (),
        *,
        n_dim: int | None = None,
        boundary: str = 'open',
        label: str = 'space',
    ) -> None:
        super().__init__(label)
        self.extent = extent
        self.n_dim = n_dim
        self.boundary = boundary

    def _check(self) -> Self:
        """Raise an error for an unknown boundary, no dimension, a side that is not positive, or walls in `R^d`."""
        extent = tuple(self.extent)
        n_dim = self.n_dim if self.n_dim is not None else len(extent)
        if self.boundary not in BOUNDARIES:
            error_msg = f'Unknown boundary {self.boundary!r}. Use one of {", ".join(BOUNDARIES)}.'
            raise ValidationError(error_msg)
        if not n_dim:
            error_msg = 'A continuous space needs the sides of its box or a number of dimensions.'
            raise ValidationError(error_msg)
        if extent and (len(extent) != n_dim or min(extent) <= 0):
            error_msg = f'The box {extent} needs one positive side for each of its {n_dim} dimensions.'
            raise ValidationError(error_msg)
        if not extent and self.boundary != 'open':
            error_msg = f'An unbounded space has no walls for its boundary {self.boundary!r}. Give the sides of a box.'
            raise ValidationError(error_msg)
        return self

    def _prepare(self, context: None = None) -> Self:
        """Set the extent of the box, its dimension, and its boundary, with no sites."""
        self.extent_ = tuple(float(side) for side in self.extent)
        self.n_dim_ = int(self.n_dim) if self.n_dim is not None else len(self.extent_)
        self.size_ = None
        self.shape_ = ()
        self.positions_ = np.zeros((0, self.n_dim_))
        self.periodic_ = self.boundary == 'periodic'
        self.default_visual_ = None
        return self

    def _get_neighbors(self, site: int, neighborhood: BaseNeighborhood | None = None) -> npt.NDArray[np.intp]:
        """Return no neighbours, since the space has no sites."""
        return np.zeros(0, dtype=np.intp)
