"""Wolf that wanders, eats a sheep of its cell, reproduces, and dies without energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ._wolf_view import WolfView


class WolfElement(Element):
    """Wolf that wanders, eats a sheep of its cell, reproduces, and dies without energy.

    In its own activation, a wolf wanders to a neighbouring cell at a cost of one unit of energy, eats and removes one
    random sheep of its cell, and dies below zero energy or gives birth with probability `reproduction`, by member
    interactions.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        energy:
            The initial energy.

        gain:
            The energy a wolf gains from a sheep.

        reproduction:
            The probability of reproducing at every step.

    Examples:
        >>> from skcomplex.core.elements import GroupElement
        >>> from skcomplex.models.biology.ecology import WolfElement
        >>> GroupElement('wolves', element=WolfElement, size=25).size
        25
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'radius': 0.35}),
        *,
        energy: Annotated[float, StateSpec()] = 20.0,
        gain: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
        reproduction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.energy = energy
        self.gain = gain
        self.reproduction = reproduction

    @classmethod
    def _get_view(cls) -> type[WolfView]:
        """Return the view of a wolf, `WolfView`."""
        return WolfView
