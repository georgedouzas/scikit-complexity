"""Define the spaces that vector states live in."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import ClassVar

from sklearn.base import BaseEstimator

from ..base import SpaceKind, ValidationError


class Space(BaseEstimator):
    """Base class of spaces, which give vector states and parameters their dimension.

    Subclasses expose the number of dimensions as `n_dim`, and name in `lacks` the options of their kind they do not
    offer, such as `'regular-geometry'` for the checkerboard schedule of a square lattice and `'image'` for image
    visuals.
    """

    space_kind: ClassVar[SpaceKind]
    lacks: ClassVar[frozenset[str]] = frozenset()

    n_dim: int


BOUNDARIES = ('open', 'periodic', 'reflecting')


class ContinuousSpace(Space):
    """Continuous space of real positions, a box with a boundary or the whole of `R^d`.

    A box has an extent along every dimension, from 0 to the extent. Its boundary acts on every position state of the
    components on it: `'open'` leaves positions alone, `'periodic'` joins opposite sides and measures pair distances by
    the minimum image, and `'reflecting'` mirrors positions back into the box and reverses the velocity across the wall.
    Without an extent the space is unbounded and has `n_dim` dimensions. Agents on a two-dimensional box move in it and
    find neighbours within a radius.

    Args:
        *sides:
            The side of the box along every dimension, or none for an unbounded space.

        extent:
            The sides as one sequence, instead of `sides`.

        n_dim:
            The number of dimensions of an unbounded space, or `None` for the number of sides of the box.

        boundary:
            `'open'`, `'periodic'`, or `'reflecting'`.

    Raises:
        ValidationError:
            The boundary is unknown, the space has neither an extent nor a dimension, a side is not positive, or an
            unbounded space has a boundary other than open.

    Examples:
        >>> from skcomplex.core.space import ContinuousSpace
        >>> ContinuousSpace(10.0, 5.0, boundary='periodic').height
        5.0
        >>> ContinuousSpace(n_dim=3).n_dim
        3
    """

    space_kind: ClassVar[SpaceKind] = SpaceKind.CONTINUOUS

    def __init__(
        self,
        *sides: float,
        extent: Sequence[float] = (),
        n_dim: int | None = None,
        boundary: str = 'open',
    ) -> None:
        given = tuple(float(side) for side in (sides or extent))
        self.extent = extent if not sides and isinstance(extent, tuple) and extent == given else given
        self.n_dim = n_dim if n_dim is not None else len(self.extent)
        self.boundary = boundary
        if boundary not in BOUNDARIES:
            error_msg = f'Unknown boundary {boundary!r}. Use one of {", ".join(BOUNDARIES)}.'
            raise ValidationError(error_msg)
        if not self.n_dim:
            error_msg = 'A continuous space needs the sides of its box or a number of dimensions.'
            raise ValidationError(error_msg)
        if self.extent and (len(self.extent) != self.n_dim or min(self.extent) <= 0):
            error_msg = f'The box {self.extent} needs one positive side for each of its {self.n_dim} dimensions.'
            raise ValidationError(error_msg)
        if not self.extent and boundary != 'open':
            error_msg = f'An unbounded space has no walls for its boundary {boundary!r}. Give the sides of a box.'
            raise ValidationError(error_msg)

    @property
    def width(self) -> float:
        """Return the side of the box along the first dimension."""
        return self.extent[0]

    @property
    def height(self) -> float:
        """Return the side of the box along the second dimension."""
        return self.extent[1]

    @property
    def torus(self) -> bool:
        """Check whether opposite sides of the box are joined."""
        return self.boundary == 'periodic'

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Return the extent, the dimension, and the boundary, so the space clones."""
        return {'extent': self.extent, 'n_dim': self.n_dim, 'boundary': self.boundary}

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the extent or the dimension, and the boundary."""
        sides = ', '.join(map(repr, self.extent)) if self.extent else f'n_dim={self.n_dim}'
        boundary = '' if self.boundary == 'open' else f', boundary={self.boundary!r}'
        return f'ContinuousSpace({sides}{boundary})'
