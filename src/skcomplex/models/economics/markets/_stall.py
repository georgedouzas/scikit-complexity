"""Market stall that sells some kinds of produce at the wholesale price plus its markup."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw
from ....specs import StateSpec, VisualSpec
from ._stall_view import StallView


class StallElement(Element):
    """Market stall that sells some kinds of produce at the wholesale price plus its markup.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        stall:
            The index of the stall.

        stock:
            The bit mask of the kinds of produce that the stall sells.

        markup:
            The relative markup on the wholesale prices.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import StallElement
        >>> GroupElement('stalls', element=StallElement, size=3, values={'stall': [0, 1, 2]}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        stall: Annotated[float, StateSpec(domain=NN)] = 0,
        stock: Annotated[float, StateSpec(domain=NN)] = 0,
        markup: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.stall = stall
        self.stock = stock
        self.markup = markup

    @classmethod
    def _get_view(cls) -> type[StallView]:
        """Return the view of a stall, with its index, stock, and markup."""
        return StallView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since stalls only wait for shoppers.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []
