"""Define the neighbours and neighbourhoods of lattices, networks, and continuous spaces, and memory sets."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Protocol, Self

from ..base import BaseNeighborhood, BaseSpace, Requires, SpaceKind, ValidationError

SITE_KINDS = frozenset({SpaceKind.LATTICE, SpaceKind.NETWORK})


def neighbor_pairs(space: BaseSpace | None, size: int, layer: str | None = None) -> list[tuple[int, int]]:
    """Return the pairs of neighbouring members of a space of sites, such as a lattice, or of a network, each pair once.

    Args:
        space:
            The space of sites or the network, fitted here unless it was fitted before.

        size:
            The number of members, one per site or node.

        layer:
            The layer of a network, or `None` for the graph of the network.

    Returns:
        pairs:
            The member indices of every pair of neighbours.

    Raises:
        ValidationError:
            The space is neither a space of sites nor a network.

    Examples:
        >>> from skcomplex.core.space import LatticeSpace, neighbor_pairs
        >>> neighbor_pairs(LatticeSpace(3, boundary='fixed'), 3)
        [(0, 1), (1, 2)]
    """
    if space is None or space.space_kind not in SITE_KINDS:
        error_msg = f'Neighbours are defined on a space of sites, such as a lattice, or on a network, not on {space!r}.'
        raise ValidationError(error_msg)
    if not hasattr(space, 'size_'):
        space.fit()
    return space._neighbor_pairs(layer)


@dataclass(frozen=True)
class Radius(BaseNeighborhood):
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

    def _check(self) -> Self:
        """Return the neighbourhood with its arguments checked and normalized."""
        changes: dict[str, Any] = {}
        if not float(self.radius) > 0:
            error_msg = f'A radius neighbourhood needs a positive radius, but got {self.radius!r}.'
            raise ValidationError(error_msg)
        changes['radius'] = float(self.radius)
        return replace(self, **changes) if changes else self


@dataclass(frozen=True)
class Nearest(BaseNeighborhood):
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

    def _check(self) -> Self:
        """Return the neighbourhood with its arguments checked and normalized."""
        changes: dict[str, Any] = {}
        if int(self.k) != self.k or int(self.k) < 1:
            error_msg = f'A nearest neighbourhood needs a whole number of neighbours of at least 1, but got {self.k!r}.'
            raise ValidationError(error_msg)
        changes['k'] = int(self.k)
        return replace(self, **changes) if changes else self


@dataclass(frozen=True)
class Cone(BaseNeighborhood):
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

    def _check(self) -> Self:
        """Return the neighbourhood with its arguments checked and normalized."""
        changes: dict[str, Any] = {}
        if not float(self.radius) > 0:
            error_msg = f'A cone neighbourhood needs a positive radius, but got {self.radius!r}.'
            raise ValidationError(error_msg)
        if not 0 < float(self.angle) <= math.pi:
            error_msg = f'A cone neighbourhood needs a half-angle in (0, pi], but got {self.angle!r}.'
            raise ValidationError(error_msg)
        changes['radius'] = float(self.radius)
        changes['angle'] = float(self.angle)
        return replace(self, **changes) if changes else self


METRIC_NEIGHBORHOODS = (Radius, Nearest, Cone)


class _Generated(Protocol):
    """Group given by the names of its generators."""

    generators: tuple[str, ...]


@dataclass(frozen=True)
class Memory(BaseNeighborhood):
    """Ordered memory set of a cellular automaton on a group space: the group elements whose values a rule reads.

    A cell at the element `g` reads the cell at `g s` for every element `s` of the memory set, in its order, so the
    neighbourhood keeps its orientation. Elements are words in the generators, such as `'a'`, `'a^-1'`, or `'a*b'`,
    and `'1'` is the identity, the cell itself.

    Args:
        elements:
            The words of the memory set.

    Examples:
        >>> from skcomplex.core.space import Memory
        >>> len(Memory(('1', 'a', 'a^-1')).elements)
        3
    """

    requires: ClassVar[Requires] = Requires(space={SpaceKind.LATTICE})

    elements: tuple[str, ...]

    def _check(self) -> Self:
        """Return the neighbourhood with its arguments checked and normalized."""
        changes: dict[str, Any] = {}
        changes['elements'] = tuple(str(element) for element in self.elements)
        return replace(self, **changes) if changes else self

    @classmethod
    def generators(
        cls,
        group: _Generated,
        *,
        center: bool = True,
        inverses: bool = True,
    ) -> 'Memory':
        """Return the memory set of the generators of a group, after the identity and with their inverses if asked.

        Args:
            group:
                The group.

            center:
                Whether the identity comes first.

            inverses:
                Whether every generator is followed by its inverse.

        Returns:
            memory:
                The memory set.

        Examples:
            >>> from skcomplex.formalisms.groups import finitely_presented_group
            >>> from skcomplex.core.space import Memory
            >>> Memory.generators(finitely_presented_group(['a', 'b'])).elements
            ('1', 'a', 'a^-1', 'b', 'b^-1')
        """
        elements = [word for name in group.generators for word in ((name, f'{name}^-1') if inverses else (name,))]
        return cls(('1', *elements) if center else tuple(elements))
