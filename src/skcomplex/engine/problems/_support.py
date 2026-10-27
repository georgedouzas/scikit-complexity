"""Declare what a solver or a backend supports: the classes of time, space, and laws, and the kinds of noise and orders."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from ._kinds import NoiseKind, OrderKind


@dataclass(frozen=True)
class SupportSpec:
    """What a solver or a backend supports: the classes of time, space, and laws, and the kinds of noise and orders.

    An empty field supports any class or kind. A class is supported when it is one of the classes of its field or a
    subclass of one, and `None` in the space field supports a system without a space.

    Args:
        time:
            The time classes supported.

        space:
            The space classes supported, with `None` for no space.

        laws:
            The law classes supported.

        noise:
            The structures of Wiener processes supported.

        orders:
            The kinds of derivative orders supported.

    Examples:
        >>> from skcomplex.core.laws import NoiseLaw, RateLaw
        >>> from skcomplex.engine.problems import SupportSpec
        >>> SupportSpec(laws={RateLaw}).missing('laws', [RateLaw, NoiseLaw])
        ['NoiseLaw']
    """

    time: AbstractSet[type] = frozenset()
    space: AbstractSet[type | None] = frozenset()
    laws: AbstractSet[type] = frozenset()
    noise: frozenset[NoiseKind] = frozenset()
    orders: frozenset[OrderKind] = frozenset()

    def _supports(self, category: str, term: object) -> bool:
        """Check whether one term of a category is supported."""
        supported = getattr(self, category)
        if not supported:
            return True
        if isinstance(term, type):
            return any(isinstance(item, type) and issubclass(term, item) for item in supported)
        return term in supported

    def missing(self, category: str, terms: Iterable[object]) -> list[str]:
        """Return the names of the terms of a category that are not supported, sorted.

        Args:
            category:
                The name of the field, such as `'laws'`.

            terms:
                The classes or kinds a system needs, with `None` for a system without a space.

        Returns:
            missing:
                The names of the unsupported terms.
        """
        found = {getattr(term, '__name__', str(term)) for term in terms if not self._supports(category, term)}
        return sorted(found)
