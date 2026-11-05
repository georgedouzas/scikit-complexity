"""Site of a lattice of rock, paper, and scissors, 0, 1, and 2, where each strategy beats the one before it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class CyclicSpeciesElement(Element):
    """Site of a lattice of rock, paper, and scissors, 0, 1, and 2, where each strategy beats the one before it.

    The sites update one at a time in random order by a `LocalRuleInteraction`, and a site adopts the strategy of a
    random neighbour that beats it.

    Args:
        label:
            The label of the site or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        strategy:
            The initial strategy, 0, 1, or 2.

    Examples:
        >>> from skcomplex.systems.biology.evolution import CyclicSpeciesElement
        >>> CyclicSpeciesElement('site', strategy=2).strategy
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        strategy: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.strategy = strategy
