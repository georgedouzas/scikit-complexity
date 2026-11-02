"""The canonical averages of a system of many levels, held by one element."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import StateSpec


class BoltzmannAverageElement(Element):
    """Log partition function, mean energy, and entropy per spin of a group of levels.

    Args:
        label:
            The label of the element.

        log_partition:
            The initial guess of the log partition function, which the equations replace.

        energy:
            The initial guess of the mean energy, which the equations replace.

        entropy:
            The initial guess of the entropy per spin, which the equations replace.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import BoltzmannAverageElement
        >>> BoltzmannAverageElement('ensemble').entropy
        0.0
    """

    def __init__(
        self,
        label: str,
        log_partition: Annotated[float, StateSpec()] = 0.0,
        energy: Annotated[float, StateSpec()] = 0.0,
        entropy: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.log_partition = log_partition
        self.energy = energy
        self.entropy = entropy
