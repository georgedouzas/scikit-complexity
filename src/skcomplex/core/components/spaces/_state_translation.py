"""Define the neighbourhood that reads a group space at a site times its own value."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ._base import BaseNeighborhood


class StateTranslationNeighborhood(BaseNeighborhood):
    """A neighbourhood that moves with the value each cell of a group space holds.

    The cell at `g` reads the cells at `g x(g) s` for every word `s`, where `x(g)` is the group element that the first
    read state holds at `g`, as the number of its site. The sites read move with the configuration, so on an infinite
    group the map has no finite memory set, as for `x(g x(g))` of Ceccherini-Silberstein and Coornaert (2023, Example
    1.8.2), which is continuous and equivariant but no cellular automaton. A run truncates it to a ball or a finite
    quotient, and a product outside the ball reads the fill value of the space.

    Unlike `MemoryNeighborhood`, whose memory set is fixed, it shifts the memory set by the value of the cell. Choose it
    when studying maps on groups that are not cellular automata.

    It appears in mathematics, in symbolic dynamics on groups, as the example of Ceccherini-Silberstein and Coornaert
    shows.

    Args:
        elements:
            The words `s` of the memory set after the translation, `'1'` for the translated site alone.

        label:
            The name of the neighbourhood, such as `'moore'`.

    Attributes:
        elements_:
            The words of the memory set as strings.

    Examples:
        >>> from skcomplex.core.components.spaces import StateTranslationNeighborhood
        >>> StateTranslationNeighborhood().prepare().elements_
        ('1',)
    """

    def __init__(self, elements: Sequence[str] = ('1',), *, label: str = 'neighborhood') -> None:
        super().__init__(label)
        self.elements = elements

    def _prepare(self, context: None = None) -> None:
        """Set `elements_`, the words of the memory set as strings."""
        super()._prepare(context)
        self.elements_ = tuple(str(element) for element in self.elements)
