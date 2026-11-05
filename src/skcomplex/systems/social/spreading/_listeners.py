"""Number of people in one state of a rumour: ignorant, spreading, or stifling."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class ListenersElement(Element):
    """Number of people in one state of a rumour: ignorant, spreading, or stifling.

    Args:
        label:
            The label of the element.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        n:
            The initial number of people.

    Examples:
        >>> from skcomplex.systems.social.spreading import ListenersElement
        >>> ListenersElement('spreaders', n=1).n
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[float, StateSpec(domain=NN)] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
