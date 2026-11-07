"""Variable of a generalised logical network, a discrete level from zero to its largest level."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class LogicalVariableElement(Element):
    """Variable of a generalised logical network, a discrete level from zero to its largest level.

    The levels stand for qualitative concentrations, such as absent, basal, medium, or high, cut at the thresholds at
    which the variable acts on its targets.

    Args:
        label:
            The label of the variable.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        level:
            The initial level.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import LogicalVariableElement
        >>> LogicalVariableElement('x', level=2.0).level
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        level: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.level = level
