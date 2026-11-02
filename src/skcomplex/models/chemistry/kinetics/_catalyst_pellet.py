"""Catalyst pellet, the concentration `c` of the reactant at its surface relative to the bulk."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class CatalystPelletElement(Element):
    """Catalyst pellet, the concentration `c` of the reactant at its surface relative to the bulk.

    The mass transfer and the surface reaction of `catalyst_pellet` change the concentration and read their constants
    from the pellet.

    Args:
        label:
            The label of the pellet.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        c:
            The initial surface concentration.

        h:
            The coefficient of mass transfer from the bulk.

        k:
            The rate constant of the surface reaction.

        K:
            The adsorption constant, which sets the inhibition.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import CatalystPelletElement
        >>> CatalystPelletElement('pellet').K
        20.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        c: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        h: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.1,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 20.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.h = h
        self.k = k
        self.K = K
