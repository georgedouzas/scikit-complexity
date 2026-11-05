"""Cell of an excitable medium, with a potential `u` and a recovery `v`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class FitzHughNagumoElement(Element):
    """Cell of an excitable medium, with a potential `u` and a recovery `v`.

    The cell has no laws of its own. FitzHugh-Nagumo kinetics and a diffusion of the potential change it.

    Args:
        label:
            The label of the cell.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial potential.

        v:
            The initial recovery.

    Examples:
        >>> from skcomplex.systems.biology.physiology import FitzHughNagumoElement
        >>> FitzHughNagumoElement('cell', u=1.5).u
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v
