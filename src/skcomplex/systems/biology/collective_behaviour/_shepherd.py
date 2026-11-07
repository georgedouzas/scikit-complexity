"""Shepherd of a flock, an agent whose mode records whether it collects stray sheep or drives the flock."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._shepherd_view import ShepherdView


class ShepherdElement(Element):
    """Shepherd of a flock, an agent whose mode records whether it collects stray sheep or drives the flock.

    Args:
        label:
            The label of the shepherd or of the group of shepherds.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        driving:
            1 while the shepherd drives the flock and 0 while it collects a stray sheep.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import ShepherdElement
        >>> ShepherdElement('dog').driving
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {}),
        *,
        driving: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.driving = driving

    @classmethod
    def _get_view(cls) -> type[ShepherdView]:
        """Return the view of the agent, `ShepherdView`."""
        return ShepherdView
