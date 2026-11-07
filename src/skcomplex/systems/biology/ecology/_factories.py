"""Ecology systems: population growth, predation, and invasion fronts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, Interaction, MapInteraction
from ....core.components.laws import BaseLaw, RateLaw, UpdateLaw
from ....core.components.laws.utils import lag, lookup, series
from ....core.components.system import BaseSystem, System
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.populations import (
    ExponentialGrowthInteraction,
    LogisticGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
    PredationInteraction,
)

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

    law_classes = (RateLaw,)

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


class RabbitsElement(Element):
    """Pairs of rabbits of this month and of last month, the two age classes of Fibonacci's recurrence.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        pairs:
            The number of pairs this month, a state that keeps its value one month earlier.

        pairs_lag1:
            The number of pairs last month.

    Examples:
        >>> from skcomplex.systems.biology.ecology import RabbitsElement
        >>> RabbitsElement('rabbits', pairs=3.0).pairs
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        pairs: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), lags=1, initial=(0.0,))] = 1.0,
        pairs_lag1: float = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.pairs = pairs
        self.pairs_lag1 = pairs_lag1


class RickerGrowthInteraction(Interaction):
    """Ricker growth of a population in discrete generations, `n' = r n exp(-n)`, overcompensating at high density.

    The size `n` is in units of the density at which the offspring of a generation peak, so a crowded generation
    leaves fewer offspring than a sparser one.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        r:
            The growth factor at low density.

    Examples:
        >>> from skcomplex.systems.biology.ecology import RickerGrowthInteraction
        >>> RickerGrowthInteraction('growth', target='fish', r=12.0).r
        12.0
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the next generation `r n exp(-n)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        n = s.target.n
        return [UpdateLaw(n, s.r * n * exp(-n))]


class SeasonalLogisticGrowthInteraction(LogisticGrowthInteraction):
    """Logistic growth under a carrying capacity that varies periodically, `K(t) = K (1 + amplitude sin(2 pi t / T))`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        r:
            The per capita growth rate at low density.

        K:
            The mean carrying capacity.

        amplitude:
            The relative amplitude of the variation of the capacity, below 1.

        period:
            The period `T` of the variation.

    Examples:
        >>> from skcomplex.systems.biology.ecology import SeasonalLogisticGrowthInteraction
        >>> SeasonalLogisticGrowthInteraction('growth', target='rabbits', period=12.0).period
        12.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
        amplitude: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        period: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, target=target, r=r, K=K)
        self.amplitude = amplitude
        self.period = period

    def _get_capacity(self, namespace: SimpleNamespace) -> Expression:
        """Return the capacity `K (1 + amplitude sin(2 pi t / T))` at the time of the namespace.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            capacity:
                The carrying capacity at the time `t`.
        """
        s = namespace
        capacity: Expression = s.K * (1 + s.amplitude * sin(2 * pi * s.t / s.period))
        return capacity


class ExponentialGrowthSystem(BaseSystem):
    """A population that grows or declines at a constant per-capita rate.

    The system has the population `population`.

    Args:
        n:
            The initial size.

        r:
            The per-capita growth rate.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ExponentialGrowthSystem
        >>> ExponentialGrowthSystem(r=0.2).prepare().system_.interactions[0].r
        0.2
    """

    field = 'biology/ecology'
    topics = ('population', 'ode')
    references = ('Malthus 1798',)
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        n: float = 10.0,
        r: float = 0.1,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r

    def _build_system(self) -> System:
        """Return the parts of a population that grows or declines at a constant per-capita rate."""
        return System(
            self.label,
            elements=[PopulationElement('population', n=self.n)],
            interactions=[ExponentialGrowthInteraction('growth', target='population', r=self.r)],
        )


class LogisticGrowthSystem(BaseSystem):
    """A population that grows logistically towards its carrying capacity.

    The system has the population `population`.

    Args:
        n:
            The initial size.

        r:
            The intrinsic growth rate.

        K:
            The carrying capacity.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> LogisticGrowthSystem(K=500.0).prepare().system_.interactions[0].K
        500.0
    """

    field = 'biology/ecology'
    topics = ('population', 'ode')
    references = ('Verhulst 1838',)
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        n: float = 10.0,
        r: float = 1.0,
        K: float = 100.0,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r
        self.K = K

    def _build_system(self) -> System:
        """Return the parts of a population that grows logistically towards its carrying capacity."""
        return System(
            self.label,
            elements=[PopulationElement('population', n=self.n)],
            interactions=[LogisticGrowthInteraction('growth', target='population', r=self.r, K=self.K)],
        )


class VaryingCapacityLogisticSystem(BaseSystem):
    """Logistic growth under a seasonal carrying capacity, which settles on a cycle of the season's period.

    Whatever its initial size, the population tends to a unique periodic solution, which lags behind the capacity and
    varies less than it. The system has the population `population`.

    Args:
        n:
            The initial size.

        r:
            The intrinsic growth rate.

        K:
            The mean carrying capacity.

        amplitude:
            The relative amplitude of the seasonal variation of the capacity, below 1.

        period:
            The period of the seasons.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import VaryingCapacityLogisticSystem
        >>> VaryingCapacityLogisticSystem(amplitude=0.2).prepare().system_.interactions[0].amplitude
        0.2
    """

    field = 'biology/ecology'
    topics = ('population', 'ode', 'forcing', 'periodic')
    references = ('Coleman 1979', 'Trzaska 2023')
    until = 20.0
    dt = 0.05

    def __init__(
        self,
        n: float = 10.0,
        *,
        r: float = 2.0,
        K: float = 100.0,
        amplitude: float = 0.5,
        period: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r
        self.K = K
        self.amplitude = amplitude
        self.period = period

    def _build_system(self) -> System:
        """Return the parts of the system."""
        return System(
            self.label,
            elements=[PopulationElement('population', n=self.n)],
            interactions=[
                SeasonalLogisticGrowthInteraction(
                    'growth',
                    target='population',
                    r=self.r,
                    K=self.K,
                    amplitude=self.amplitude,
                    period=self.period,
                ),
            ],
        )


class HutchinsonSystem(BaseSystem):
    """Hutchinson's delayed logistic equation, which oscillates around `K` when `r * tau` exceeds π/2.

    The history before the start is the initial size. The system has the population `population`.

    Args:
        n:
            The initial size, also its history.

        r:
            The per-capita growth rate at low density.

        K:
            The carrying capacity.

        tau:
            The delay of the density feedback.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import HutchinsonSystem
        >>> HutchinsonSystem(tau=1.0).prepare().system_.interactions[0].tau
        1.0
    """

    field = 'biology/ecology'
    topics = ('population', 'delay', 'oscillation')
    references = ('Hutchinson 1948', 'Erneux 2009')
    until = 100.0
    dt = 0.1

    def __init__(
        self,
        n: float = 50.0,
        *,
        r: float = 1.0,
        K: float = 100.0,
        tau: float = 2.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r
        self.K = K
        self.tau = tau

    def _build_system(self) -> System:
        """Return the parts of the system."""
        return System(
            self.label,
            elements=[PopulationElement('population', n=self.n)],
            interactions=[LogisticGrowthInteraction('growth', target='population', r=self.r, K=self.K, tau=self.tau)],
        )


class LotkaVolterraSystem(BaseSystem):
    """Prey and predators whose populations oscillate through predation.

    The system has the populations `prey` and `predators`.

    Args:
        prey:
            The initial number of prey.

        predators:
            The initial number of predators.

        r:
            The growth rate of the prey.

        a:
            The rate at which predation removes prey.

        b:
            The rate at which predation feeds predators.

        d:
            The mortality rate of the predators.

        K:
            The carrying capacity of the prey, or `None` for exponential growth of the prey.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraSystem
        >>> [element.label for element in LotkaVolterraSystem().prepare().system_.elements]
        ['prey', 'predators']
    """

    field = 'biology/ecology'
    topics = ('predator-prey', 'ode', 'oscillation')
    references = ('Lotka 1925', 'Volterra 1926')
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        prey: float = 40.0,
        predators: float = 9.0,
        *,
        r: float = 1.1,
        a: float = 0.4,
        b: float = 0.1,
        d: float = 0.4,
        K: float | None = None,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.prey = prey
        self.predators = predators
        self.r = r
        self.a = a
        self.b = b
        self.d = d
        self.K = K

    def _build_system(self) -> System:
        """Return the parts of Prey and predators whose populations oscillate through predation."""
        growth: BaseInteraction = (
            ExponentialGrowthInteraction('growth', target='prey', r=self.r)
            if self.K is None
            else LogisticGrowthInteraction('growth', target='prey', r=self.r, K=self.K)
        )
        return System(
            self.label,
            elements=[PopulationElement('prey', n=self.prey), PopulationElement('predators', n=self.predators)],
            interactions=[
                growth,
                PredationInteraction('predation', predator='predators', prey='prey', a=self.a, b=self.b),
                MortalityInteraction('death', target='predators', d=self.d),
            ],
        )


class FibonacciRabbitsSystem(BaseSystem):
    """Fibonacci's rabbits, `N_{t+1} = N_t + N_{t-1}`, where every pair older than one month breeds a new pair.

    The previous population counts the pairs old enough to breed, so the recurrence is an age structure of two classes.
    The system has the population `rabbits.pairs` and last month's `rabbits.pairs_lag1`.

    Args:
        pairs:
            The number of pairs this month.

        previous:
            The number of pairs last month.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import FibonacciRabbitsSystem
        >>> FibonacciRabbitsSystem().prepare().system_.elements[0].pairs
        1.0
    """

    field = 'biology/ecology'
    topics = ('population', 'discrete map', 'second order', 'age structure')
    references = ('Fibonacci 1202',)
    until = 20
    dt = 1

    def __init__(
        self,
        pairs: float = 1.0,
        previous: float = 0.0,
        *,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.pairs = pairs
        self.previous = previous

    def _build_system(self) -> System:
        """Return the parts of the system."""
        breeding = MapInteraction(
            'breeding',
            target='rabbits',
            state='pairs',
            function=lambda s: s.target.pairs + lag(s.target.pairs, 1),
        )
        return System(
            self.label,
            elements=[RabbitsElement('rabbits', pairs=self.pairs, pairs_lag1=self.previous)],
            interactions=[breeding],
        )


class RickerSystem(BaseSystem):
    """Ricker's map of a population in discrete generations, `n' = r n exp(-n)`, chaotic for large `r`.

    The positive fixed point `n = ln r` is stable while `1 < r < e^2`. Beyond `e^2`, about 7.39, it loses stability to a
    two-cycle, and further period doublings lead to chaos near `r = 14.8`. The size stays positive for every `r`. The
    system has the population `population.n`.

    Args:
        n:
            The initial size, in units of the density at which the offspring peak.

        r:
            The growth factor at low density.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import RickerSystem
        >>> RickerSystem(r=12.0).prepare().system_.interactions[0].r
        12.0
    """

    field = 'biology/ecology'
    topics = ('population', 'discrete map', 'period doubling', 'chaos')
    references = ('Ricker 1954', 'Dilão 2023')
    until = 100
    dt = 1

    def __init__(
        self,
        n: float = 0.5,
        *,
        r: float = 2.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n = n
        self.r = r

    def _build_system(self) -> System:
        """Return the parts of the system."""
        return System(
            self.label,
            elements=[PopulationElement('population', n=self.n)],
            interactions=[RickerGrowthInteraction('growth', target='population', r=self.r)],
        )


class PenaeidShrimpSystem(BaseSystem):
    """The growth of one penaeid shrimp from the juvenile stage, forced by the food and the water temperature.

    The individual ingests food up to a maximum ration set by its biomass and the water temperature, assimilates its
    gut content, respires, and spawns, while its gonads grow and its carapace length follows von Bertalanffy's curve
    `L(t) = L_inf (1 - exp(-K t))` from zero, which nears its asymptote after six to eight months. The gut content
    settles near `I / A_e`, the ingestion over the assimilation efficiency.

    The coefficients are those of the listing of Bala et al. (2017). With them the respiration of a juvenile at 28 C,
    `respiration exp(0.07 T) (phi B)^0.185`, is about half its biomass per day, more than its maximum ration, so the
    total biomass declines slowly while the gonads and the length grow. The weight gain the source describes needs a
    lower `respiration` coefficient, such as 0.02. The system has the states `shrimp.gut`, `shrimp.biomass`,
    `shrimp.gonads`, and `shrimp.length`.

    Args:
        gut:
            The initial gut content, in grams of dry weight.

        biomass:
            The initial total biomass, in grams of dry weight.

        ration:
            The factor of the maximum ration, lower for males.

        respiration:
            The coefficient of the weight-specific respiration, per day.

        L_inf:
            The asymptotic carapace length, in millimetres.

        K:
            The growth coefficient of the length, per day.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import PenaeidShrimpSystem
        >>> PenaeidShrimpSystem().prepare().system_.elements[0].K
        0.013
    """

    field = 'biology/ecology'
    topics = ('individual growth', 'bioenergetics', 'aquaculture', 'stocks and flows', 'ode')
    references = ('Franco et al. 2006', 'Bala et al. 2017')
    until = 60.0
    dt = 0.5

    def __init__(
        self,
        gut: float = 0.5,
        biomass: float = 1.0,
        *,
        ration: float = 2.5,
        respiration: float = 0.05,
        L_inf: float = 29.9,
        K: float = 0.013,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.gut = gut
        self.biomass = biomass
        self.ration = ration
        self.respiration = respiration
        self.L_inf = L_inf
        self.K = K

    def _build_system(self) -> System:
        """Return the parts of the system."""
        return System(
            self.label,
            elements=[
                ShrimpElement(
                    'shrimp',
                    gut=self.gut,
                    biomass=self.biomass,
                    ration=self.ration,
                    respiration=self.respiration,
                    L_inf=self.L_inf,
                    K=self.K,
                ),
            ],
        )
