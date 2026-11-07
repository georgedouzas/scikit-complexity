"""Resident of a population with a fixed resistance to a disease, immune for good once infected."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class ResidentElement(Element):
    """Resident of a population with a fixed resistance to a disease, immune for good once infected.

    Args:
        label:
            The label of the resident or of the group of residents.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        R:
            The resistance, between 0 and 1, where 1 is immunity.

        infected:
            Whether the resident was infected in the last outbreak.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ResidentElement
        >>> ResidentElement('resident', R=0.3).R
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        R: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.5,
        infected: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.R = R
        self.infected = infected
