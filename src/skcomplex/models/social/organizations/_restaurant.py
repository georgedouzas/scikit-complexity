"""Restaurant of the obesity model: a market at 0, a restaurant at 1, or a fast food outlet at 2."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec
from ._restaurant_view import RestaurantView


class RestaurantElement(Element):
    """Restaurant of the obesity model: a market at 0, a restaurant at 1, or a fast food outlet at 2.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        kind:
            The kind of restaurant.

        sold:
            The number of meals sold.

        sold_high:
            The number of meals sold from the top half of the restaurant's calorie range.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.social.organizations import RestaurantElement
        >>> group = GroupElement('restaurants', element=RestaurantElement, size=3, values={'kind': 2.0})
        >>> group.label, group.element.__name__, group.size
        ('restaurants', 'RestaurantElement', 3)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        kind: Annotated[float, StateSpec(domain=NN)] = 0.0,
        sold: Annotated[float, StateSpec(domain=NN)] = 0.0,
        sold_high: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kind = kind
        self.sold = sold
        self.sold_high = sold_high

    @classmethod
    def _get_view(cls) -> type[RestaurantView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return RestaurantView
