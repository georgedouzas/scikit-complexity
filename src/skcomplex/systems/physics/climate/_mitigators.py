"""Share of mitigators in a population that weighs the cost of mitigation against the perceived cost of warming."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class MitigatorsElement(Element):
    """Share of mitigators in a population that weighs the cost of mitigation against the perceived cost of warming.

    The share `x` follows the replicator equation `x' = x (1 - x) (-alpha + f + delta (2 x - 1))`, with the cost of
    mitigation `alpha`, the strength `delta` of the social norm, and the cost of warming
    `f = fmax / (1 + exp(-w (Tf - Tc)))`. The perceived warming `Tf = tf / tp (T(t) - T(t - tp))` projects the warming
    of the last `tp` years over `tf` years. A rate interaction reads the temperature it perceives, so the element holds
    the state and the parameters it reads.

    Args:
        label:
            The label of the population.

        visual:
            How the population is drawn, or `None` for `default_visual_`.

        x:
            The initial share of mitigators.

        alpha:
            The cost of mitigation.

        delta:
            The strength of the social norm.

        fmax:
            The largest cost of warming.

        w:
            The steepness of the cost of warming.

        Tc:
            The critical warming.

        tp:
            The years of past warming that the projection reads.

        tf:
            The years ahead of the projection.

    Attributes:
        default_visual_:
            The visual of the population when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.climate import MitigatorsElement
        >>> MitigatorsElement('people').tp
        10.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.5,
        alpha: Annotated[float, ParameterSpec()] = 1.0,
        delta: Annotated[float, ParameterSpec()] = 1.0,
        fmax: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        w: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
        Tc: Annotated[float, ParameterSpec()] = 1.0,
        tp: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
        tf: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.alpha = alpha
        self.delta = delta
        self.fmax = fmax
        self.w = w
        self.Tc = Tc
        self.tp = tp
        self.tf = tf
