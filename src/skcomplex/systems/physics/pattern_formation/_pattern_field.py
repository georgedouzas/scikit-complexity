"""Site of a real field `u` whose patterns a free energy selects."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class PatternFieldElement(Element):
    """Site of a real field `u` whose patterns a free energy selects.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the site is drawn, or `None` for `default_visual_`.

        u:
            The initial value of the field.

    Attributes:
        default_visual_:
            The visual of the site when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.pattern_formation import PatternFieldElement
        >>> PatternFieldElement('site', u=0.1).u
        0.1
    """

    def __init__(self, label: str, visual: VisualSpec | None = None, *, u: Annotated[float, StateSpec()] = 0.0) -> None:
        super().__init__(label, visual=visual)
        self.u = u
