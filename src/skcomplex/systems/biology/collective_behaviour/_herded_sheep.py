"""Sheep of a herded flock, an agent with a heading that a shepherding interaction moves."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec
from ._herded_sheep_view import HerdedSheepView


class HerdedSheepElement(Element):
    """Sheep of a herded flock, an agent with a heading that a shepherding interaction moves.

    Args:
        label:
            The label of the sheep or of the flock.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        hx:
            The first component of the heading of the last move.

        hy:
            The second component of the heading of the last move.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import HerdedSheepElement
        >>> HerdedSheepElement('sheep', hx=1.0).hx
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {}),
        *,
        hx: Annotated[float, StateSpec()] = 0.0,
        hy: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.hx = hx
        self.hy = hy

    @classmethod
    def _get_view(cls) -> type[HerdedSheepView]:
        """Return the view of the agent, `HerdedSheepView`."""
        return HerdedSheepView
