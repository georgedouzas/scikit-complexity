"""Vegetation cover of a tropical forest, whose growth rate falls on both sides of an optimal temperature."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ForestCoverElement(Element):
    """Vegetation cover of a tropical forest, whose growth rate falls on both sides of an optimal temperature.

    The cover `v` runs from 0 for bare soil to 1 for full forest. Bare soil is warmer than forest by `a`, so the local
    temperature is `T = Tv + a (1 - v)`, and the growth rate `g = g0 (1 - ((T - Topt) / beta)²)` peaks at `Topt`. A
    rate interaction gives the logistic growth `g v (1 - v)` and the disturbance `gamma v`, so the element holds the
    state and the parameters it reads.

    Args:
        label:
            The label of the forest.

        visual:
            How the forest is drawn, or `None` for `default_visual_`.

        v:
            The initial vegetation cover.

        Tv:
            The ambient temperature, the temperature under full forest cover, in degrees Celsius.

        g0:
            The largest growth rate.

        beta:
            The half width of the growth rate against the temperature.

        Topt:
            The optimal temperature of growth.

        a:
            The warming of bare soil over forest.

        gamma:
            The disturbance rate.

    Attributes:
        default_visual_:
            The visual of the forest when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.climate import ForestCoverElement
        >>> ForestCoverElement('forest').Topt
        28.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.8,
        Tv: Annotated[float, ParameterSpec()] = 30.0,
        g0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        Topt: Annotated[float, ParameterSpec()] = 28.0,
        a: Annotated[float, ParameterSpec()] = 5.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.Tv = Tv
        self.g0 = g0
        self.beta = beta
        self.Topt = Topt
        self.a = a
        self.gamma = gamma
