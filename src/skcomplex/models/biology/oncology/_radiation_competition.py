"""Normal and tumour cells in Lotka-Volterra competition under periodic irradiation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class RadiationCompetitionElement(Element):
    """Normal and tumour cells in Lotka-Volterra competition under periodic irradiation.

    The normal cells follow `dN/dt = r1 N (1 - b1 N) - c1 N A - epsilon N` and the tumour cells
    `dA/dt = r2 A (1 - b2 A) - c2 N A - a2 D(t) A`, where the irradiation `D` is 1 for `width` time units every `period`
    and 0 otherwise, and `epsilon N` is the radiation that spills over to the normal tissue.

    The element holds the states and parameters and has no laws of its own: the interactions of
    `radiotherapy_competition` read its parameters.

    Args:
        label:
            The label of the tissue.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        N:
            The initial concentration of normal cells.

        A:
            The initial concentration of tumour cells.

        r1:
            The growth rate of the normal cells.

        r2:
            The growth rate of the tumour cells.

        b1:
            The reciprocal carrying capacity of the normal cells.

        b2:
            The reciprocal carrying capacity of the tumour cells.

        c1:
            The rate at which the tumour cells suppress the normal cells.

        c2:
            The rate at which the normal cells suppress the tumour cells.

        a2:
            The per capita rate at which the radiation kills tumour cells during an irradiation.

        epsilon:
            The per capita rate at which the spilled radiation kills normal cells.

        period:
            The time between the starts of irradiations.

        width:
            The duration of an irradiation.

    Examples:
        >>> from skcomplex.models.biology.oncology import RadiationCompetitionElement
        >>> RadiationCompetitionElement('tissue', N=0.5, A=0.8).A
        0.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        N: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        A: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        r1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.008,
        r2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        b1: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        b2: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        c1: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        c2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.15,
        a2: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.001,
        period: float = 7.0,
        width: float = 5.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.N = N
        self.A = A
        self.r1 = r1
        self.r2 = r2
        self.b1 = b1
        self.b2 = b2
        self.c1 = c1
        self.c2 = c2
        self.a2 = a2
        self.epsilon = epsilon
        self.period = period
        self.width = width
