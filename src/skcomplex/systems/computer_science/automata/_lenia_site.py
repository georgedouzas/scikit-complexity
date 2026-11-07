"""Site of a Lenia world, with a value in `[0, 1]` for each of up to three channels."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class LeniaSiteElement(Element):
    """Site of a Lenia world, with a value in `[0, 1]` for each of up to three channels.

    The updates of the world change the channels, and a world of one channel leaves `B` and `C` at 0.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        A:
            The initial value of the first channel.

        B:
            The initial value of the second channel.

        C:
            The initial value of the third channel.

    Examples:
        >>> from skcomplex.systems.computer_science.automata import LeniaSiteElement
        >>> LeniaSiteElement('site', A=0.5).A
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        A: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        B: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        C: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.A = A
        self.B = B
        self.C = C
