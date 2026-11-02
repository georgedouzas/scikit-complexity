"""Till of the car wash, which counts the revenue and the cars served and turned away."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw
from ....specs import StateSpec, VisualSpec
from ._till_view import TillView


class TillElement(Element):
    """Till of the car wash, which counts the revenue and the cars served and turned away.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        revenue:
            The money taken.

        washes:
            The number of cars that left after a wash only.

        valets:
            The number of cars that left after a valet.

        turned_away:
            The number of drivers who did not turn in because of the queue.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.markets import TillElement
        >>> GroupElement('till', element=TillElement, size=1).size
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        revenue: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        washes: Annotated[float, StateSpec(domain=NN)] = 0,
        valets: Annotated[float, StateSpec(domain=NN)] = 0,
        turned_away: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.revenue = revenue
        self.washes = washes
        self.valets = valets
        self.turned_away = turned_away

    @classmethod
    def _get_view(cls) -> type[TillView]:
        """Return the view of the till, with its revenue and counts."""
        return TillView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the car wash books the payments.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []
