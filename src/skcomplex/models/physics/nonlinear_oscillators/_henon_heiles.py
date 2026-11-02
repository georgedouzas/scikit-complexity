"""Star in the Hénon-Heiles potential `V = (x^2 + y^2) / 2 + x^2 y - y^3 / 3`, with the canonical momenta `px, py`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import StateSpec, VisualSpec


class HenonHeilesElement(Element):
    """Star in the Hénon-Heiles potential `V = (x^2 + y^2) / 2 + x^2 y - y^3 / 3`, with the canonical momenta `px, py`.

    The coordinates and the momenta are states of order 1, which a Hamiltonian interaction moves. Below the escape
    energy `1 / 6` the motion is regular for small energies and mostly chaotic near the escape energy.

    Args:
        label:
            The label of the star.

        visual:
            How the star is drawn.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        px:
            The initial first momentum.

        py:
            The initial second momentum.

    Examples:
        >>> from skcomplex.models.physics.nonlinear_oscillators import HenonHeilesElement
        >>> HenonHeilesElement('star').px
        0.35
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('bodies', {'positions': ('x', 'y'), 'trail': None}),
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.1,
        px: Annotated[float, StateSpec()] = 0.35,
        py: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.px = px
        self.py = py
