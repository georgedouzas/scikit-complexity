"""Define state translation neighbourhoods.

They read a group space at the product of a site and its own value.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from ._base import BaseNeighborhood


class StateTranslationNeighborhood(BaseNeighborhood):
    """Memory set translated by the value of a cell.

    The cell at `g` reads the cells at `g x(g) s` for every word `s`. The alphabet is the group of a group space, and a
    cell holds a group element as the number of its site. A local rule over the neighbourhood reads, at every site `g`,
    the cells at the products `g x(g) s`, where `x(g)` is the value of the first state the rule reads. The sites read
    move with the configuration, so on an infinite group the map has no finite memory set, as for `x(g · x(g))` of
    Ceccherini-Silberstein and Coornaert (2023, Example 1.8.2), which is continuous and equivariant but no cellular
    automaton. A run truncates it to a ball or a finite quotient: on a finite group every product is a site, and on a
    ball a product outside it, or reached through a site outside it, reads the fill value of the space, as does a value
    that names no site.

    Args:
        elements:
            The words `s` of the memory set after the translation, `'1'` for the translated site alone.

        label:
            The label of the neighbourhood.

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
        self.elements_ = tuple(str(element) for element in self.elements)
