"""Define the continuous space that position states live in."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ..base import BaseSpace, SpaceKind, ValidationError

BOUNDARIES = ('open', 'periodic', 'reflecting')


class ContinuousSpace(BaseSpace):
    """Continuous space of real positions, a box with a boundary or the whole of `R^d`.

    A box has an extent along every dimension, from 0 to the extent. Its boundary acts on every position state of the
    components on it: `'open'` leaves positions alone, `'periodic'` joins opposite sides and measures pair distances by
    the minimum image, and `'reflecting'` mirrors positions back into the box and reverses the velocity across the wall.
    Without an extent the space is unbounded and has `n_dim` dimensions. Agents on a two-dimensional box move in it and
    find neighbours within a radius. The positions are states of the members, so the space has no sites, and a site
    has no neighbours in it.

    Args:
        *sides:
            The side of the box along every dimension, or none for an unbounded space.

        extent:
            The sides as one sequence, instead of `sides`.

        n_dim:
            The number of dimensions of an unbounded space, or `None` for the number of sides of the box.

        boundary:
            `'open'`, `'periodic'`, or `'reflecting'`.

    Attributes:
        extent_:
            The side of the box along every dimension, or `()` for an unbounded space.

    Examples:
        >>> from skcomplex.core.space import ContinuousSpace
        >>> ContinuousSpace(10.0, 5.0, boundary='periodic').fit().extent_
        (10.0, 5.0)
        >>> ContinuousSpace(n_dim=3).fit().n_dim_
        3
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.CONTINUOUS

    extent_: tuple[float, ...]

    def __init__(
        self,
        *sides: float,
        extent: Sequence[float] = (),
        n_dim: int | None = None,
        boundary: str = 'open',
    ) -> None:
        self.sides = sides
        self.extent = extent
        self.n_dim = n_dim
        self.boundary = boundary

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the extent, the dimension, and the boundary, so the space clones.

        Args:
            deep:
                Unused, since the arguments hold no estimators.

        Returns:
            params:
                The sides given as one extent, the dimension, and the boundary.
        """
        return {'extent': tuple(self.sides or self.extent), 'n_dim': self.n_dim, 'boundary': self.boundary}

    def _check(self) -> None:
        """Raise an error for an unknown boundary, a missing dimension, a side that is not positive, or walls in `R^d`.

        Raises:
            ValidationError:
                The boundary is unknown, the space has neither an extent nor a dimension, a side is not positive, or an
                unbounded space has a boundary other than open.
        """
        extent = tuple(self.sides or self.extent)
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

    def _fit(self) -> None:
        """Set the extent of the box, its dimension, and its boundary, with no sites."""
        self.extent_ = tuple(float(side) for side in (self.sides or self.extent))
        self.n_dim_ = int(self.n_dim) if self.n_dim is not None else len(self.extent_)
        self.size_ = None
        self.shape_ = ()
        self.positions_ = np.zeros((0, self.n_dim_))
        self.periodic_ = self.boundary == 'periodic'
        self.default_visual_ = None

    def _run(self, site: int, neighborhood: Any = None) -> npt.NDArray[np.intp]:  # noqa: ANN401
        """Return no neighbours, since the space has no sites."""
        return np.zeros(0, dtype=np.intp)
