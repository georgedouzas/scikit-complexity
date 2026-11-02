"""A word of a language, with its cost and its probability."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import StateSpec


class WordElement(Element):
    """Word with a quenched cost and the probability of its use.

    Args:
        label:
            The label of the word or the group.

        cost:
            The cost of the word.

        probability:
            The initial guess of the probability, which the equation replaces.

    Examples:
        >>> from skcomplex.models.social.language import WordElement
        >>> WordElement('word', cost=1.0).cost
        1.0
    """

    def __init__(
        self,
        label: str,
        cost: Annotated[float, StateSpec()] = 1.0,
        probability: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.cost = cost
        self.probability = probability
