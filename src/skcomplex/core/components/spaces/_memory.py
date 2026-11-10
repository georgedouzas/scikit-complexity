"""Define the neighbourhood of the memory set of a cellular automaton on a group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ._base import BaseNeighborhood


class MemoryNeighborhood(BaseNeighborhood):
    """A neighbourhood of the group elements whose values a cellular automaton reads.

    The memory set is a list of elements of a group. A cell at the element `g` reads the cell at `g s` for every element
    `s` of the memory set, in its order, so the neighbourhood keeps its orientation. Elements are words in the
    generators, such as `'a'`, `'a^-1'`, or `'a*b'`, and `'1'` is the identity, the cell itself.

    Unlike `LatticeNeighborhood`, whose offsets are integer vectors, it takes words of a group. Choose it when cells
    live on a `CayleySpace`.

    It appears in mathematics and computer science, in cellular automata on free groups and on groups of automata.

    Args:
        elements:
            The words of the memory set.

        label:
            The name of the neighbourhood, such as `'moore'`.

    Attributes:
        elements_:
            The words of the memory set as strings.

    Examples:
        >>> from skcomplex.core.components.spaces import MemoryNeighborhood
        >>> MemoryNeighborhood(['1', 'a', 'a^-1']).prepare().elements_
        ('1', 'a', 'a^-1')
    """

    def __init__(self, elements: Sequence[str], *, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.elements = elements

    def _get_elements(self) -> tuple[str, ...]:
        """Return the words of the memory set as strings."""
        return tuple(str(element) for element in self.elements)

    def _prepare(self, context: None = None) -> None:
        """Set `elements_`, the words of the memory set as strings."""
        super()._prepare(context)
        self.elements_ = self._get_elements()
