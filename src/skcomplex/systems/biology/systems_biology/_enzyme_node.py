"""Enzyme of a signalling circuit, the active fraction `X` of a pool that other enzymes activate and deactivate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class EnzymeNodeElement(Element):
    """Enzyme of a signalling circuit, the active fraction `X` of a pool that other enzymes activate and deactivate.

    Args:
        label:
            The label of the enzyme.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        X:
            The initial active fraction.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import EnzymeNodeElement
        >>> EnzymeNodeElement('A', X=0.2).X
        0.2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        X: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.X = X
