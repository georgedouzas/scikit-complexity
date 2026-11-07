"""Penaeid shrimp, an individual whose gut, biomass, gonads, and length grow on food at the water temperature."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import lookup, series
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

DAYS = (0.0, 29.2, 58.3, 87.5, 117.0, 146.0, 175.0, 204.0, 233.0, 263.0, 292.0, 321.0, 350.0)
FOOD = (695.0, 750.0, 750.0, 730.0, 720.0, 710.0, 710.0, 735.0, 685.0, 630.0, 665.0, 720.0, 710.0)
TEMPERATURE = (27.3, 29.5, 30.5, 26.5, 27.3, 28.8, 31.5, 29.5, 25.8, 27.5, 27.8, 29.5, 29.8)
ALLOMETRY = (
    (1.0, 0.15),
    (1.9, 0.243),
    (2.8, 0.327),
    (3.7, 0.393),
    (4.6, 0.474),
    (5.5, 0.543),
    (6.4, 0.615),
    (7.3, 0.663),
    (8.2, 0.708),
    (9.1, 0.735),
    (10.0, 0.75),
)
THERMAL = (
    (25.0, 0.902),
    (25.7, 0.92),
    (26.4, 0.951),
    (27.1, 0.99),
    (27.8, 1.02),
    (28.5, 1.06),
    (29.2, 1.08),
    (29.9, 1.08),
    (30.6, 1.09),
    (31.3, 1.10),
    (32.0, 1.10),
)


class ShrimpElement(Element):
    """Penaeid shrimp, an individual whose gut, biomass, gonads, and length grow on food at the water temperature.

    The shrimp ingests `I = R_max (1 - exp(-k F))` of the food `F`, with the maximum ration `R_max` the product of a
    table of its biomass, a table of the water temperature, and the factor `ration`. Its gut content `G` follows
    `dG/dt = I - A_e G`, and its total biomass `dB/dt = A_e G - M - S`, with the respiration `M = a exp(0.07 T) (phi
    B)^0.185 B` and the spawning `S = 9.3 L^2.8 E_w` of eggs of weight `E_w`. Its gonads grow as `dB_g/dt = GSI B`, and
    its carapace length follows von Bertalanffy, `dL/dt = K (L_inf - L)`. The food and the water temperature are the
    yearly series of the source, interpolated linearly in days.

    Args:
        label:
            The label of the shrimp.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        gut:
            The initial gut content, in grams of dry weight.

        biomass:
            The initial total biomass, in grams of dry weight.

        gonads:
            The initial reproductive biomass, in grams of dry weight.

        length:
            The initial carapace length, in millimetres.

        ration:
            The factor of the maximum ration.

        k:
            The consumption efficiency, per milligram of dry weight per square metre.

        assimilation:
            The assimilation efficiency of the gut content, per day.

        respiration:
            The coefficient of the weight-specific respiration, per day.

        phi:
            The conversion factor from dry to fresh weight.

        gsi:
            The gonadosomatic index, the growth of the gonads per unit biomass per day.

        egg_weight:
            The dry weight of one egg, in grams.

        L_inf:
            The asymptotic carapace length, in millimetres.

        K:
            The growth coefficient of the length, per day.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ShrimpElement
        >>> ShrimpElement('shrimp').L_inf
        29.9
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        gut: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        biomass: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gonads: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0001,
        length: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        ration: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.5,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.12,
        assimilation: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.8,
        respiration: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        phi: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        gsi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        egg_weight: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0e-6,
        L_inf: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 29.9,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.013,
    ) -> None:
        super().__init__(label, visual=visual)
        self.gut = gut
        self.biomass = biomass
        self.gonads = gonads
        self.length = length
        self.ration = ration
        self.k = k
        self.assimilation = assimilation
        self.respiration = respiration
        self.phi = phi
        self.gsi = gsi
        self.egg_weight = egg_weight
        self.L_inf = L_inf
        self.K = K

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the ingestion, assimilation, respiration, spawning, gonad growth, and length growth.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        food = series(s.t, DAYS, FOOD)
        temperature = series(s.t, DAYS, TEMPERATURE)
        max_ration = lookup(s.biomass, ALLOMETRY) * lookup(temperature, THERMAL) * s.ration
        ingestion = max_ration * (1 - exp(-s.k * food))
        assimilated = s.gut * s.assimilation
        respired = s.respiration * exp(0.07 * temperature) * (s.biomass * s.phi) ** 0.185 * s.biomass
        spawning = 9.3 * s.length**2.8 * s.egg_weight
        return [
            RateLaw(s.gut, ingestion - assimilated),
            RateLaw(s.biomass, assimilated - respired - spawning),
            RateLaw(s.gonads, s.biomass * s.gsi),
            RateLaw(s.length, s.K * (s.L_inf - s.length)),
        ]
