"""Sheep that wanders, eats grown grass, reproduces, and dies without energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._sheep_view import SheepView


class SheepElement(Element):
    """Sheep that wanders, eats grown grass, reproduces, and dies without energy.

    In its own activation, a sheep wanders to a neighbouring cell at a cost of one unit of energy, eats the grown grass
    of its cell, and dies below zero energy or gives birth with probability `reproduction`, splitting its energy with
    the lamb, by member interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        energy:
            The initial energy.

        gain:
            The energy a sheep gains from grown grass.

        reproduction:
            The probability of reproducing at every step.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.ecology import SheepElement
        >>> GroupElement('sheep', element=SheepElement, size=100).size
        100
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.3}),
        *,
        energy: Annotated[float, StateSpec()] = 8.0,
        gain: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        reproduction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.04,
    ) -> None:
        super().__init__(label, visual=visual)
        self.energy = energy
        self.gain = gain
        self.reproduction = reproduction

    @classmethod
    def _get_view(cls) -> type[SheepView]:
        """Return the view of a sheep, `SheepView`."""
        return SheepView
