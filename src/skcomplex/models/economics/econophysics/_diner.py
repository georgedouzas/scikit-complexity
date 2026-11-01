"""Diner of the Kolkata Paise Restaurant problem, who goes to one restaurant every evening."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._diner_view import DinerView


class DinerElement(Element):
    """Diner of the Kolkata Paise Restaurant problem, who goes to one restaurant every evening.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        restaurant:
            The restaurant chosen last, from 0 to the number of restaurants minus one.

        crowd:
            The number of diners at that restaurant, 0 before the first evening.

        served:
            Whether the diner got a meal, 1 or 0.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.economics.econophysics import DinerElement
        >>> GroupElement('diners', element=DinerElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        restaurant: Annotated[float, StateSpec(domain=NN)] = 0,
        crowd: Annotated[float, StateSpec(domain=NN)] = 0,
        served: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.restaurant = restaurant
        self.crowd = crowd
        self.served = served

    @classmethod
    def _get_view(cls) -> type[DinerView]:
        """Return the view of the agents of the element."""
        return DinerView
