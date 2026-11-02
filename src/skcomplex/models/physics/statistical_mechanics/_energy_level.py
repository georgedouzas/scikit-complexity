"""An energy level of a disordered system, a quenched state of a group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import StateSpec


class EnergyLevelElement(Element):
    """Energy level of a configuration, held fixed.

    Args:
        label:
            The label of the level or the group.

        energy:
            The energy of the level.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import EnergyLevelElement
        >>> EnergyLevelElement('level', energy=-1.5).energy
        -1.5
    """

    def __init__(self, label: str, energy: Annotated[float, StateSpec()] = 0.0) -> None:
        super().__init__(label)
        self.energy = energy
