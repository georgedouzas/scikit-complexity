"""Site of a lattice with a static electric charge."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, VisualSpec


class StaticChargeElement(Element):
    """Site of a lattice with a static electric charge `q`.

    Args:
        label:
            The label of the site or the group on the sites.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        q:
            The electric charge.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.models.physics.waves import StaticChargeElement
        >>> StaticChargeElement('site', q=-1.0).q
        -1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        q: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.q = q
