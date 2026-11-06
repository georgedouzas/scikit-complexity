"""Site of a field `u` that the Burgers equation `du/dt + d(u^2 / 2)/dx = 0` carries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class BurgersElement(Element):
    """Site of a field `u` that the Burgers equation `du/dt + d(u^2 / 2)/dx = 0` carries.

    Args:
        label:
            The label of the site.

        visual:
            How the site is drawn, or `None` for none.

        u:
            The initial value.

    Examples:
        >>> from skcomplex.systems.physics.fluids import BurgersElement
        >>> BurgersElement('site', u=1.5).u
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
