"""A cell of a plain at a distance from the market, with the crop grown on it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec


class LandCellElement(Element):
    """Cell of a plain that grows one crop, numbered from 1, or lies idle as crop 0.

    Args:
        label:
            The label of the cell or the field.

        crop:
            The initial guess of the crop, which the equation replaces.

        distance:
            The distance from the market.

    Examples:
        >>> from skcomplex.systems.economics.resources import LandCellElement
        >>> LandCellElement('cell', distance=3.0).distance
        3.0
    """

    def __init__(
        self,
        label: str,
        crop: Annotated[float, StateSpec()] = 0.0,
        distance: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.crop = crop
        self.distance = distance
