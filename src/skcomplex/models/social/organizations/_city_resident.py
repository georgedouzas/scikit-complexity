"""A resident of a city at a fixed position, with the sum of the interactions with all the others."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from ....core.elements import Element
from ....specs import StateSpec


class CityResidentElement(Element):
    """Resident of a city, who stays at a position.

    Args:
        label:
            The label of the resident or the group.

        x:
            The position.

        contacts:
            The initial guess of the sum of the interactions with the others, which the equation replaces.

    Examples:
        >>> from skcomplex.models.social.organizations import CityResidentElement
        >>> CityResidentElement('resident', x=(1.0, 2.0)).x
        (1.0, 2.0)
    """

    def __init__(
        self,
        label: str,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, position=True)] = 0.0,
        contacts: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.contacts = contacts
