"""Define the neighbourhood of the members within a distance, and within a cone."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import math

from ...errors import ValidationError
from ...types import IntTuple
from ._base import BaseNeighborhood

METRICS = ('euclidean', 'chebyshev', 'manhattan', 'hexagonal')
HEXAGONAL_DIRECTIONS = ((0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1))
HEXAGONAL_DIMENSIONS = 2


class RadiusNeighborhood(BaseNeighborhood):
    """A neighbourhood of the members within a distance under a metric.

    In a continuous space, it is the members whose positions lie within `radius`, by the nearest periodic image in a
    periodic box. With an `angle` below `pi`, a neighbour must also lie in the cone around the direction of the member,
    so `angle = pi` is the whole disk and `angle = pi / 2` the half ahead. The direction is a vector state of the
    members, or in two dimensions a scalar state that holds the angle.

    On a lattice, it is the offsets within `radius` under the metric, in as many dimensions as the lattice has:

    - `'euclidean'`: the length of the offset is at most the radius.
    - `'chebyshev'`: every coordinate of the offset is at most the radius. For radius 1 in two dimensions these are the
      8 cells of the Moore neighbourhood.
    - `'manhattan'`: the sum of the absolute coordinates is at most the radius. For radius 1 in two dimensions these are
      the 4 cells of the von Neumann neighbourhood.
    - `'hexagonal'`: `max(|a|, |b|, |a + b|)` is at most the radius on a triangular lattice, in two dimensions. The
      offsets come ring by ring, each counterclockwise, and for radius 1 direction `k` points at `60 k` degrees.

    The other metrics give the offsets in lexicographic order. With `center`, the site itself comes first.

    Unlike `NearestNeighborhood`, which takes a fixed number of members, it takes every member within a distance, so the
    number of neighbours varies. Unlike `LatticeNeighborhood`, which takes its offsets one by one, it builds them from a
    radius and a metric. Choose it when members sense others up to a range, or when a law reads the usual neighbours on
    a grid.

    It appears in many fields. Birds align with flockmates within sight in the Vicsek model, atoms interact within a
    cutoff, a sandpile topples grains to its four neighbours, and a cell of the Game of Life counts its eight
    neighbours.

    Args:
        radius:
            The largest distance of a neighbour.

        metric:
            `'euclidean'`, `'chebyshev'`, `'manhattan'`, or `'hexagonal'`. A continuous space measures euclidean
            distances.

        angle:
            The half-angle of the cone, between 0 and `pi`, which is the whole disk. A lattice takes the whole disk.

        direction:
            The name of the state that holds the direction, or `None` for the velocity, the vector state declared as the
            rate of the position. Only a cone, with an angle below `pi`, reads it.

        center:
            Whether a member is its own neighbour, the first one on a lattice.

        label:
            The name of the neighbourhood, such as `'moore'`.

    Examples:
        >>> import math
        >>> from skcomplex.core.components.spaces import RadiusNeighborhood
        >>> RadiusNeighborhood(1, metric='chebyshev').prepare().metric
        'chebyshev'
        >>> RadiusNeighborhood(1.0, angle=math.pi / 2, direction='theta').direction
        'theta'
    """

    def __init__(
        self,
        radius: float,
        *,
        metric: str = 'euclidean',
        angle: float = math.pi,
        direction: str | None = None,
        center: bool = False,
        label: str = 'neighborhood',
    ) -> None:
        super().__init__(label)
        self.radius = radius
        self.metric = metric
        self.angle = angle
        self.direction = direction
        self.center = center

    def _get_hexagonal_offsets(self) -> tuple[IntTuple, ...]:
        """Return the offsets of a triangular lattice within the radius, ring by ring.

        Each ring runs counterclockwise, starts at the radius of the ring along the second axis, and walks its six
        sides. The site itself comes first.
        """
        offsets: list[IntTuple] = [(0, 0)]
        for ring in range(1, math.floor(self.radius) + 1):
            first, second = 0, ring
            for side in range(6):
                step_first, step_second = HEXAGONAL_DIRECTIONS[(side + 2) % 6]
                for _ in range(ring):
                    offsets.append((first, second))
                    first, second = first + step_first, second + step_second
        return tuple(offsets)

    def _is_within(self, offset: IntTuple) -> bool:
        """Check whether an offset lies within the radius under the metric."""
        steps = [abs(step) for step in offset]
        if self.metric == 'chebyshev':
            return max(steps, default=0) <= self.radius
        if self.metric == 'manhattan':
            return sum(steps) <= self.radius
        return sum(step**2 for step in steps) <= self.radius**2

    def _get_offsets(self, n_dim: int | None = None) -> tuple[IntTuple, ...] | None:
        """Return the offsets within the radius on a lattice of some dimensions.

        Args:
            n_dim:
                The number of dimensions of the lattice, or `None` when no lattice is known yet.

        Returns:
            offsets:
                The offsets in their order, or `None` without a number of dimensions, unless the hexagonal metric
                fixes it to two.

        Raises:
            ValidationError:
                The neighbourhood is a cone, or the hexagonal metric is used outside two dimensions.
        """
        self.prepare()
        if self.metric == 'hexagonal' and n_dim is None:
            n_dim = HEXAGONAL_DIMENSIONS
        if n_dim is None:
            return None
        if float(self.angle) < math.pi:
            error_msg = f'The radius neighbourhood {self.label!r} is a cone, which a lattice cannot read. Use angle=pi.'
            raise ValidationError(error_msg)
        if self.metric == 'hexagonal' and n_dim != HEXAGONAL_DIMENSIONS:
            error_msg = f'The hexagonal neighbourhood {self.label!r} lives in two dimensions, not {n_dim}.'
            raise ValidationError(error_msg)
        if self.metric == 'hexagonal':
            offsets = self._get_hexagonal_offsets()
        else:
            reach = range(-math.floor(self.radius), math.floor(self.radius) + 1)
            offsets = tuple(offset for offset in itertools.product(reach, repeat=n_dim) if self._is_within(offset))
        site = (0,) * n_dim
        others = tuple(offset for offset in offsets if offset != site)
        return (site, *others) if self.center else others

    def _prepare(self, context: None = None) -> None:
        """Raise an error for an unknown metric or centre, or a bad radius or half-angle.

        The radius must be positive, and the half-angle in `(0, pi]`.
        """
        super()._prepare(context)
        if self.metric not in METRICS:
            error_msg = f'A radius neighbourhood has the metric {self.metric!r}. Use one of {", ".join(METRICS)}.'
            raise ValidationError(error_msg)
        if not isinstance(self.center, bool):
            error_msg = f'A radius neighbourhood has the centre {self.center!r}. Use True or False.'
            raise ValidationError(error_msg)
        if not float(self.radius) > 0:
            error_msg = f'A radius neighbourhood needs a positive radius, but got {self.radius!r}.'
            raise ValidationError(error_msg)
        if not 0 < float(self.angle) <= math.pi:
            error_msg = f'A radius neighbourhood needs a half-angle in (0, pi], but got {self.angle!r}.'
            raise ValidationError(error_msg)
