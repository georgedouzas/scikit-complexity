"""Prey that walks at random and gives birth in cells it does not share with other prey."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, VisualSpec
from ._prey_view import PreyView


class PreyElement(Element):
    """Prey that walks at random and gives birth in cells it does not share with other prey.

    In its own activation a prey walks to a random neighbouring cell and then gives birth with the probability
    `birth` when no other prey shares the new cell, by member interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        birth:
            The probability of giving birth at every step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.ecology import PreyElement
        >>> GroupElement('rabbits', element=PreyElement, size=200).size
        200
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.3}),
        *,
        birth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.birth = birth

    @classmethod
    def _get_view(cls) -> type[PreyView]:
        """Return the view of a prey, `PreyView`."""
        return PreyView
