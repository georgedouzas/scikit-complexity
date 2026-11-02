"""A candidate solution of a search, a point of a box."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy.typing as npt

from ....core.elements import Element
from ....specs import StateSpec


class SearchCandidateElement(Element):
    """Candidate solution that holds a vector.

    Args:
        label:
            The label of the candidate or the group.

        x:
            The vector of the candidate.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import SearchCandidateElement
        >>> SearchCandidateElement('candidate', x=(0.0, 1.0)).x
        (0.0, 1.0)
    """

    def __init__(self, label: str, x: Annotated[npt.ArrayLike, StateSpec(vector=True)] = (0.0, 0.0)) -> None:
        super().__init__(label)
        self.x = x
