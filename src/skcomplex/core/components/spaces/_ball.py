"""Define ball neighbourhoods, the lattice offsets within a radius under a metric."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools

from ...errors import ValidationError
from ...types import IntTuple
from ._base import BaseNeighborhood
from ._lattice_neighborhood import LatticeNeighborhood

METRICS = ('chebyshev', 'manhattan', 'hexagonal')
CENTERS = (None, 'first', 'inline')
HEXAGONAL_DIRECTIONS = ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))
HEXAGONAL_DIMENSIONS = 2


class BallNeighborhood(LatticeNeighborhood):
    """Lattice offsets within a radius under a metric.

    The Chebyshev metric bounds every coordinate of an offset by the radius, and the Manhattan metric bounds the sum of
    their absolute values. Both give the offsets in lexicographic order. In two dimensions and for radius 1 they are the
    8 cells of the Moore neighbourhood and the 4 cells of the von Neumann neighbourhood. The hexagonal metric bounds
    `max(|a|, |b|, |a + b|)` on a two-dimensional triangular lattice and gives the offsets ring by ring, each ring
    counterclockwise from the direction of the second axis. For radius 1 direction `k` points at `60 k` degrees in the
    plane, so direction `k + 3` modulo 6 is opposite to direction `k`.

    Args:
        radius:
            The largest distance of an offset from the site.

        metric:
            `'chebyshev'`, `'manhattan'`, or `'hexagonal'`.

        n_dim:
            The number of dimensions, 2 for the hexagonal metric.

        center:
            Where the site itself comes: `None` leaves it out, `'first'` puts it first, and `'inline'` keeps it in its
            lexicographic place, the order of Wolfram's rule numbers in one dimension.

        label:
            The label of the neighbourhood.

    Attributes:
        offsets_:
            The offsets as tuples of integers.

    Examples:
        >>> from skcomplex.core.components.spaces import BallNeighborhood
        >>> len(BallNeighborhood().prepare().offsets_)
        8
        >>> BallNeighborhood(metric='manhattan').prepare().offsets_
        ((-1, 0), (0, -1), (0, 1), (1, 0))
        >>> BallNeighborhood(n_dim=1, center='inline').prepare().offsets_
        ((-1,), (0,), (1,))
        >>> BallNeighborhood(metric='hexagonal').prepare().offsets_
        ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))
    """

    def __init__(
        self,
        radius: int = 1,
        *,
        metric: str = 'chebyshev',
        n_dim: int = 2,
        center: str | None = None,
        label: str = 'neighborhood',
    ) -> None:
        BaseNeighborhood.__init__(self, label)
        self.radius = radius
        self.metric = metric
        self.n_dim = n_dim
        self.center = center

    def _get_chebyshev_offsets(self) -> tuple[IntTuple, ...]:
        """Return every offset whose coordinates are all within the radius, in lexicographic order."""
        return tuple(itertools.product(range(-self.radius, self.radius + 1), repeat=self.n_dim))

    def _get_manhattan_offsets(self) -> tuple[IntTuple, ...]:
        """Return every offset whose absolute coordinates sum to at most the radius, in lexicographic order."""
        return tuple(offset for offset in self._get_chebyshev_offsets() if sum(map(abs, offset)) <= self.radius)

    def _get_hexagonal_offsets(self) -> tuple[IntTuple, ...]:
        """Return the offsets of a triangular lattice within the radius, ring by ring and counterclockwise.

        Every ring starts at the radius of the ring along the second axis, and walks its six sides.
        """
        offsets: list[IntTuple] = [(0, 0)]
        for ring in range(1, self.radius + 1):
            first, second = 0, ring
            for side in range(6):
                step_first, step_second = HEXAGONAL_DIRECTIONS[(side + 2) % 6]
                for _ in range(ring):
                    offsets.append((first, second))
                    first, second = first + step_first, second + step_second
        return tuple(offsets)

    def _get_offsets(self) -> tuple[IntTuple, ...]:
        """Return the offsets of the ball, with the site itself where `center` places it."""
        offsets = {
            'chebyshev': self._get_chebyshev_offsets,
            'manhattan': self._get_manhattan_offsets,
            'hexagonal': self._get_hexagonal_offsets,
        }[self.metric]()
        site = (0,) * self.n_dim
        if self.center == 'inline':
            return offsets
        others = tuple(offset for offset in offsets if offset != site)
        return (site, *others) if self.center == 'first' else others

    def _check(self) -> None:
        """Raise an error for a metric, a centre, a radius, or a dimension the ball cannot take."""
        if self.metric not in METRICS:
            error_msg = f'The ball {self.label!r} has the metric {self.metric!r}. Use one of {", ".join(METRICS)}.'
            raise ValidationError(error_msg)
        if self.center not in CENTERS:
            error_msg = f"The ball {self.label!r} has the centre {self.center!r}. Use None, 'first', or 'inline'."
            raise ValidationError(error_msg)
        if self.radius < 0:
            error_msg = f'The ball {self.label!r} has the radius {self.radius!r}. Give a nonnegative radius.'
            raise ValidationError(error_msg)
        if self.metric == 'hexagonal' and self.n_dim != HEXAGONAL_DIMENSIONS:
            error_msg = f'The hexagonal ball {self.label!r} has {self.n_dim} dimensions. It lives in two dimensions.'
            raise ValidationError(error_msg)
        if self.metric == 'hexagonal' and self.center == 'inline':
            error_msg = (
                f"The hexagonal ball {self.label!r} keeps its site 'inline', a lexicographic place it does not have. "
                "Use None or 'first'."
            )
            raise ValidationError(error_msg)

    def _prepare(self, context: None = None) -> None:
        """Set `offsets_`, the offsets of the ball.

        It first raises an error for an unknown metric or centre, a negative radius, a hexagonal ball outside two
        dimensions, or a hexagonal ball with the site inline.
        """
        self._check()
        super()._prepare(context)
