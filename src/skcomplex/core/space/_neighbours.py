"""List the neighbouring members of lattices and networks, and define the neighbourhoods of continuous spaces."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..base import Requires, SpaceKind, ValidationError
from ._cayley import GroupSpace
from ._lattices import LatticeSpace
from ._networks import NetworkSpace
from ._spaces import Space


def neighbour_pairs(space: Space | None, size: int, layer: str | None = None) -> list[tuple[int, int]]:
    """Return the pairs of neighbouring members of a lattice, a group space, or a network, each pair once.

    Args:
        space:
            The lattice or the network.

        size:
            The number of members, one per site or node.

        layer:
            The layer of a network, or `None` for the graph of the network.

    Returns:
        pairs:
            The member indices of every pair of neighbours.

    Raises:
        ValidationError:
            The space is neither a lattice nor a network.

    Examples:
        >>> from skcomplex.core.space import LatticeSpace, neighbour_pairs
        >>> neighbour_pairs(LatticeSpace(3, boundary='fixed'), 3)
        [(0, 1), (1, 2)]
    """
    if isinstance(space, NetworkSpace):
        sources, targets = space.list_edges(layer)
        return [
            (first, second) for first, second in zip(sources.tolist(), targets.tolist(), strict=True) if first != second
        ]
    if isinstance(space, GroupSpace):
        found_edges = {
            frozenset((first, second)): (first, second) for first, second, _ in space.edges if first != second
        }
        return list(found_edges.values())
    if not isinstance(space, LatticeSpace):
        error_msg = f'Neighbours are defined on a LatticeSpace, a GroupSpace, or a NetworkSpace, not on {space!r}.'
        raise ValidationError(error_msg)
    dimensions = space.dimensions
    sites = np.arange(size).reshape(dimensions)
    found: dict[frozenset[int], tuple[int, int]] = {}
    for axis, length in enumerate(dimensions):
        shifted = np.roll(sites, -1, axis=axis)
        pairs = np.stack([sites.ravel(), shifted.ravel()], axis=1)
        if space.boundary != 'periodic':
            edge = np.indices(dimensions)[axis].ravel() == length - 1
            pairs = pairs[~edge]
        for first, second in pairs.tolist():
            if first != second:
                found.setdefault(frozenset((first, second)), (first, second))
    return list(found.values())


@dataclass(frozen=True)
class Radius:
    """Neighbourhood of the members of a continuous space within a distance, the metric neighbourhood.

    Distances are measured between the position states of the members, by the minimum image in a periodic box.

    Args:
        radius:
            The largest distance of a neighbour.

        center:
            Whether a member is its own neighbour.

    Raises:
        ValidationError:
            The radius is not positive.

    Examples:
        >>> from skcomplex.core.space import Radius
        >>> Radius(1.5)
        Radius(radius=1.5, center=False)
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.CONTINUOUS})

    radius: float
    center: bool = False

    def __post_init__(self) -> None:
        if not float(self.radius) > 0:
            error_msg = f'A radius neighbourhood needs a positive radius, but got {self.radius!r}.'
            raise ValidationError(error_msg)
        object.__setattr__(self, 'radius', float(self.radius))


@dataclass(frozen=True)
class Nearest:
    """Neighbourhood of the `k` members of a continuous space closest to a member, the topological neighbourhood.

    Distances are measured between the position states of the members, by the minimum image in a periodic box. A
    group of at most `k` other members gives every member all the others.

    Args:
        k:
            The number of neighbours.

        center:
            Whether a member is its own neighbour, besides its `k` nearest others.

    Raises:
        ValidationError:
            The number of neighbours is below 1.

    Examples:
        >>> from skcomplex.core.space import Nearest
        >>> Nearest(7).k
        7
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.CONTINUOUS})

    k: int
    center: bool = False

    def __post_init__(self) -> None:
        if int(self.k) != self.k or int(self.k) < 1:
            error_msg = f'A nearest neighbourhood needs a whole number of neighbours of at least 1, but got {self.k!r}.'
            raise ValidationError(error_msg)
        object.__setattr__(self, 'k', int(self.k))


@dataclass(frozen=True)
class Cone:
    """Neighbourhood of the members of a continuous space within a distance and inside a cone around a heading.

    A neighbour is within the radius of a member and its direction from the member makes an angle of at most `angle`
    with the member's heading, so `angle = pi` is the whole disk and `angle = pi / 2` the half ahead. The heading is a
    vector state of the members, or in two dimensions a scalar state that holds the heading angle.

    Args:
        radius:
            The largest distance of a neighbour.

        angle:
            The half-angle of the cone, between 0 and `pi`.

        heading:
            The name of the state that holds the heading, or `None` for the velocity, the vector state declared as
            the rate of the position.

        center:
            Whether a member is its own neighbour.

    Raises:
        ValidationError:
            The radius is not positive or the angle is outside `(0, pi]`.

    Examples:
        >>> import math
        >>> from skcomplex.core.space import Cone
        >>> Cone(1.0, math.pi / 2, heading='theta').heading
        'theta'
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.CONTINUOUS})

    radius: float
    angle: float
    heading: str | None = None
    center: bool = False

    def __post_init__(self) -> None:
        if not float(self.radius) > 0:
            error_msg = f'A cone neighbourhood needs a positive radius, but got {self.radius!r}.'
            raise ValidationError(error_msg)
        if not 0 < float(self.angle) <= math.pi:
            error_msg = f'A cone neighbourhood needs a half-angle in (0, pi], but got {self.angle!r}.'
            raise ValidationError(error_msg)
        object.__setattr__(self, 'radius', float(self.radius))
        object.__setattr__(self, 'angle', float(self.angle))


METRIC_NEIGHBOURHOODS = (Radius, Nearest, Cone)
