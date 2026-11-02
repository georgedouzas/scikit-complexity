"""A synaptic weight of a binary perceptron, held as a Boolean state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.elements import Element
from ....specs import StateSpec


class BinaryWeightElement(Element):
    """Weight of a binary perceptron, where the state 1 stands for `+1` and the state 0 for `-1`.

    Args:
        label:
            The label of the weight or the group.

        on:
            Whether the weight is `+1`.

    Examples:
        >>> from skcomplex.models.computer_science.neural_networks import BinaryWeightElement
        >>> BinaryWeightElement('weight', on=1.0).on
        1.0
    """

    def __init__(self, label: str, on: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0) -> None:
        super().__init__(label)
        self.on = on
