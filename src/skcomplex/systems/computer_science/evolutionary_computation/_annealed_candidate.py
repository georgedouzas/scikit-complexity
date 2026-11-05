"""A candidate solution of simulated annealing, a point of the plane with one state per coordinate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec


class AnnealedCandidateElement(Element):
    """Candidate solution of the plane, whose coordinates are sampled one state at a time.

    Args:
        label:
            The label of the candidate or the group.

        x:
            The first coordinate.

        y:
            The second coordinate.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import AnnealedCandidateElement
        >>> AnnealedCandidateElement('candidate', x=1.0).x
        1.0
    """

    def __init__(
        self,
        label: str,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
