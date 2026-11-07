"""Axon on an edge of a network, whose conduction delay myelination changes."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class AxonElement(Element):
    """Axon on an edge of a network, whose conduction delay myelination changes.

    Args:
        label:
            The label of the axon or of the group on the edges.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        tau:
            The initial conduction delay.

        tau0:
            The baseline delay that the delay relaxes to.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import AxonElement
        >>> AxonElement('axon', tau0=2.0).tau0
        2.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        tau: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        tau0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.tau = tau
        self.tau0 = tau0
