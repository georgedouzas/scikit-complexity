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

    A law that reads it sees every member within `radius` of a member, so the number of neighbours varies, such as the
    eight cells around a cell in `RadiusNeighborhood(1, metric='chebyshev')`. In a continuous space the distance runs
    between positions, across the walls of a periodic box, and an `angle` below `pi` keeps only the members in a cone
    ahead of the member. On a lattice the neighbours are the offsets within the radius under the metric, and with
    `center` the site itself comes first. Birds align with the flockmates they can see, atoms push on the atoms within
    a cutoff, a sandpile topples grains onto its four neighbours, and a cell of the Game of Life counts its eight.

    Read more in the user guide, under Spaces.

    Args:
        radius:
            The largest distance of a neighbour.

        metric:
            How a lattice measures the length of an offset. With `'euclidean'` it is the straight length. With
            `'chebyshev'` it is the largest coordinate, so radius 1 in two dimensions gives the 8 cells around a cell.
            With `'manhattan'` it is the sum of the coordinates, so radius 1 gives the 4 cells beside it. With
            `'hexagonal'` it is `max(|a|, |b|, |a + b|)` on a triangular lattice, and the offsets come ring by ring,
            counterclockwise. A continuous space measures straight lengths.

        angle:
            The half-angle of the cone, between 0 and `pi`, which is the whole disk. A lattice takes the whole disk.

        direction:
            The name of the state that holds the direction, or `None` for the velocity, the vector state declared as the
            rate of the position. Only a cone, with an angle below `pi`, reads it.

        center:
            Whether a member is its own neighbour, the first one on a lattice.

        label:
            The name of the neighbourhood, such as `'moore'`.

    See Also:
        NearestNeighborhood: Picks a fixed number of the closest members.
        LatticeNeighborhood: Takes its offsets one by one.
        DelaunayNeighborhood: Picks the members that touch, whatever the distance.

    Examples:
        >>> import math
        >>> from skcomplex.core.components.spaces import RadiusNeighborhood
        >>> RadiusNeighborhood(1.0, angle=math.pi / 2).prepare().angle
        1.5707963267948966
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
