"""Slow temperature anomaly of Hasselmann's stochastic climate, with its feedback rate and weather noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ClimateAnomalyElement(Element):
    """Slow temperature anomaly of Hasselmann's stochastic climate, with its feedback rate and weather noise.

    The relaxation of `hasselmann` damps the anomaly at the rate `lam` and drives it with weather noise of amplitude
    `sigma`.

    Args:
        label:
            The label of the anomaly.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        T:
            The initial temperature anomaly.

        lam:
            The feedback rate that damps the anomaly.

        sigma:
            The amplitude of the weather noise.

    Examples:
        >>> from skcomplex.systems.physics.climate import ClimateAnomalyElement
        >>> ClimateAnomalyElement('ocean').lam
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        T: Annotated[float, StateSpec()] = 0.0,
        lam: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, visual=visual)
        self.T = T
        self.lam = lam
        self.sigma = sigma
