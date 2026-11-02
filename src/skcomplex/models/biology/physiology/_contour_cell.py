"""Logistic cell of a closed contour that takes up the signalling molecules of the next cell through a power map."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class ContourCellElement(Element):
    """Logistic cell of a closed contour that takes up the signalling molecules of the next cell through a power map.

    `ContourUptakeInteraction` iterates `x_i <- (1 - c_i) r x_i (1 - x_i) + c_i x_{i+1}^{p_i}`, where the logistic map
    is the intracellular dynamics, `c_i` the share of the exchange with the environment between the cells, and `p_i`
    the affinity of the cell for the uptake. The last cell of the contour takes up from the first.

    Args:
        label:
            The label of the group of cells.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial concentration of signalling molecules in the cell, between 0 and 1.

        r:
            The logistic parameter, the disposition of the environment to communication.

        c:
            The coupling of the cell to the next one.

        p:
            The affinity of the cell for the uptake of signalling molecules.

    Examples:
        >>> from skcomplex.models.biology.physiology import ContourCellElement
        >>> ContourCellElement('cells').r
        2.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.5,
        c: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.r = r
        self.c = c
        self.p = p
