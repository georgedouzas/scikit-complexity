"""Cow of a herder that gains the energy of the grass it eats, burns its metabolism, and starves without food."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw
from ....core.specs import (
    StateSpec,
    VisualSpec,
)
from ._cow_view import CowView


class CowElement(Element):
    """Cow of a herder that gains the energy of the grass it eats, burns its metabolism, and starves without food.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        owner:
            The id of the herder that owns the cow, given initially as its position in the group `herders`.

        energy:
            The energy of the cow.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.resources import CowElement
        >>> GroupElement('cows', element=CowElement, size=3, values={'energy': 4.0}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'energy'}),
        *,
        owner: Annotated[float, StateSpec(reference='herders')] = -1.0,
        energy: Annotated[float, StateSpec()] = 4.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.owner = owner
        self.energy = energy

    @classmethod
    def _get_view(cls) -> type[CowView]:
        """Return the view of the agents of the element."""
        return CowView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the grazing moves the cows and their energy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []
