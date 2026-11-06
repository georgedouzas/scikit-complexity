"""Rider who waits at a cell for a cab to a destination, and pays the fare when the cab picks it up."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import (
    StateSpec,
    VisualSpec,
)
from ._rider_view import RiderView


class RiderElement(Element):
    """Rider who waits at a cell for a cab to a destination, and pays the fare when the cab picks it up.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        destination_x:
            The column of the destination.

        destination_y:
            The row of the destination.

        waited:
            The number of steps the rider has waited.

        seats:
            The seats the rider still requests, one until a cab is on its way and zero after.

        money:
            The money of the rider.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.engineering.traffic import RiderElement
        >>> GroupElement('riders', element=RiderElement, size=2, values={'money': 50.0}).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'waited'}),
        *,
        destination_x: Annotated[float, StateSpec(domain=NN)] = 0.0,
        destination_y: Annotated[float, StateSpec(domain=NN)] = 0.0,
        waited: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        seats: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        money: Annotated[float, StateSpec()] = 100.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.destination_x = destination_x
        self.destination_y = destination_y
        self.waited = waited
        self.seats = seats
        self.money = money

    @classmethod
    def _get_view(cls) -> type[RiderView]:
        """Return the view of the riders, which types their states."""
        return RiderView
