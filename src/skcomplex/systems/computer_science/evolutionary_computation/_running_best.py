"""The lowest value a search has seen."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec


class RunningBestElement(Element):
    """Lowest objective value of a search so far.

    Args:
        label:
            The label of the element.

        best:
            The value before the search starts.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import RunningBestElement
        >>> RunningBestElement('record', best=5.0).best
        5.0
    """

    def __init__(self, label: str, best: Annotated[float, StateSpec()] = 1e300) -> None:
        super().__init__(label)
        self.best = best
