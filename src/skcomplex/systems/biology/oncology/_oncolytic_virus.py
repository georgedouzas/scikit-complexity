"""Uninfected and infected tumour cells that share a carrying capacity, with infection by mass action."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class OncolyticVirusElement(Element):
    """Uninfected and infected tumour cells that share a carrying capacity, with infection by mass action.

    In units of the carrying capacity and of the growth time of uninfected cells, the uninfected cells follow
    `dAU/dt = AU (1 - AU - AV) - alpha AU AV` and the infected cells `dAV/dt = r AV (1 - AU - AV) + alpha AU AV - d AV`.

    The element holds the states and parameters and has no laws of its own: the interactions of `oncolytic_virus` read
    its parameters.

    Args:
        label:
            The label of the tumour.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        AU:
            The initial uninfected tumour cells.

        AV:
            The initial infected tumour cells.

        alpha:
            The infection rate.

        r:
            The growth rate of infected cells.

        d:
            The death rate of infected cells.

    Examples:
        >>> from skcomplex.systems.biology.oncology import OncolyticVirusElement
        >>> OncolyticVirusElement('tumour').alpha
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        AU: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        AV: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.AU = AU
        self.AV = AV
        self.alpha = alpha
        self.r = r
        self.d = d
