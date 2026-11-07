"""Synapse on an edge of a network, with an efficacy that plasticity changes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import StateSpec, VisualSpec


class SynapseElement(Element):
    """Synapse on an edge of a network, with an efficacy that plasticity changes.

    Args:
        label:
            The label of the synapse or of the group on the edges.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        kappa:
            The initial efficacy.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import SynapseElement
        >>> SynapseElement('synapse', kappa=0.5).kappa
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        kappa: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.kappa = kappa
