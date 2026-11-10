"""Define the support spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from ..types import NoiseKind, OrderKind
from ._base_spec import BaseSpec
from ._tags import Tags


@dataclass(frozen=True)
class SupportSpec(BaseSpec):
    """A spec of what a solver or a backend supports.

    It holds the classes of time, space, and laws, and the kinds of noise and orders. An empty field supports any class
    or kind. A class is supported when it is one of the classes of its field or a subclass of one, and `None` in the
    space field supports a system without a space.

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
        >>> from skcomplex.core.components.laws import NoiseLaw, RateLaw
        >>> from skcomplex.core.specs import SupportSpec
        >>> SupportSpec(laws={RateLaw}).get_missing('laws', [RateLaw, NoiseLaw])
        ['NoiseLaw']
        >>> sorted(SupportSpec(laws={RateLaw, NoiseLaw}).get_tags().laws)
        ['noise', 'rate']
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

    def get_missing(self, category: str, terms: Iterable[object]) -> list[str]:
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

    @staticmethod
    def _get_class_tags(classes: AbstractSet[type | None], attribute: str) -> frozenset[str]:
        """Return the tags that supported classes and their subclasses name.

        They name them in a class variable, and `None` gives `'none'`.

        Args:
            classes:
                The supported classes, with `None` for a system without a space.

            attribute:
                The class variable that names the tag or the tags of a class, such as `'law_tag'`.

        Returns:
            tags:
                The tags of the classes, of the subclasses they support, and `'none'` when `None` is supported.
        """
        found = {'none'} if None in classes else set()
        stack = [item for item in classes if item is not None]
        while stack:
            klass = stack.pop()
            value = getattr(klass, attribute, ())
            found |= {value} if isinstance(value, str) else set(value)
            stack += klass.__subclasses__()
        return frozenset(found - {''})

    def get_tags(self) -> Tags:
        """Return the capability tags supported.

        They are those of the supported classes of time, space, and laws. A supported class supports its subclasses, so
        the tags of the subclasses count too, and an empty field of the support gives an empty field of the tags, which
        supports any value.

        Returns:
            tags:
                The supported times, spaces, and law kinds.
        """
        return Tags(
            time=self._get_class_tags(self.time, 'time_tag'),
            space=self._get_class_tags(self.space, 'space_tags'),
            laws=self._get_class_tags(self.laws, 'law_tag'),
            randomness=None,
        )
