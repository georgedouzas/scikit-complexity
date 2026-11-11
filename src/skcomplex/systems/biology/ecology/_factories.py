"""Ecology systems: population growth, predation, and invasion fronts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo, pi, sin
from sage.sets.real_set import RealSet

from ....core.components.interactions import BaseInteraction, Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import lag, series
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import DeclaredVariableSpec, ParameterSpec, RoleSpec, StateSpec
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations
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
    (1.0, 1.9, 2.8, 3.7, 4.6, 5.5, 6.4, 7.3, 8.2, 9.1, 10.0),
    (0.15, 0.243, 0.327, 0.393, 0.474, 0.543, 0.615, 0.663, 0.708, 0.735, 0.75),
)
THERMAL = (
    (25.0, 25.7, 26.4, 27.1, 27.8, 28.5, 29.2, 29.9, 30.6, 31.3, 32.0),
    (0.902, 0.92, 0.951, 0.99, 1.02, 1.06, 1.08, 1.08, 1.09, 1.1, 1.1),
)


class RickerGrowthInteraction(Interaction):
    """Ricker growth of a population in discrete generations, `x' = r x exp(-x)`.

    It overcompensates at high density.

    The size `x` is in units of the density at which the offspring of a generation peak, so a crowded generation
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
        """Return the next generation `r x exp(-x)`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        n = s.target.x
        return [UpdateLaw(n, s.r * n * exp(-n))]


class SeasonalLogisticGrowthInteraction(LogisticGrowthInteraction):
    """Logistic growth under a carrying capacity that varies periodically.

    The capacity is `K(t) = K (1 + amplitude sin(2 pi t / T))`.

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
        """Return the capacity at the time of the namespace.

        It is `K (1 + amplitude sin(2 pi t / T))`.

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

    def _build_system(self) -> CompositeSystem:
        """Return the components of a population that grows at a constant per-capita rate.

        The population grows or declines.
        """
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('population', x=self.n)],
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

    def _build_system(self) -> CompositeSystem:
        """Return the components of a population that grows logistically.

        It grows towards its carrying capacity.
        """
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('population', x=self.n)],
            interactions=[LogisticGrowthInteraction('growth', target='population', r=self.r, K=self.K)],
        )


class VaryingCapacityLogisticSystem(BaseSystem):
    """Logistic growth under a seasonal carrying capacity.

    It settles on a cycle of the season's period.

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

    def _build_system(self) -> CompositeSystem:
        """Return the components of logistic growth under a seasonal carrying capacity."""
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('population', x=self.n)],
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
    """Hutchinson's delayed logistic equation.

    It oscillates around `K` when `r * tau` exceeds π/2.

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

    def _build_system(self) -> CompositeSystem:
        """Return the components of Hutchinson's delayed logistic equation."""
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('population', x=self.n)],
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

    def _build_system(self) -> CompositeSystem:
        """Return the components of prey and predators whose populations oscillate.

        Their populations oscillate through predation.
        """
        growth: BaseInteraction = (
            ExponentialGrowthInteraction('growth', target='prey', r=self.r)
            if self.K is None
            else LogisticGrowthInteraction('growth', target='prey', r=self.r, K=self.K)
        )
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('prey', x=self.prey), PopulationElement('predators', x=self.predators)],
            interactions=[
                growth,
                PredationInteraction('predation', predator='predators', prey='prey', a=self.a, b=self.b),
                MortalityInteraction('death', target='predators', d=self.d),
            ],
        )


class FibonacciRabbitsSystem(DynamicalSystem):
    """Fibonacci's rabbits, `N_{t+1} = N_t + N_{t-1}`.

    Every pair older than one month breeds a new pair.

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
        >>> FibonacciRabbitsSystem().prepare().system_.elements[0].get_value('pairs')
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
        super().__init__(label=label)
        self.pairs = pairs
        self.previous = previous

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'rabbits'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'pairs': DeclaredVariableSpec(
                self.pairs,
                StateSpec(domain=RealSet.closed_open(0, oo), lags=1, initial=(0.0,)),
            ),
        }

    def _get_equations(self) -> Equations:
        """Return the components of Fibonacci's rabbits."""
        return {
            'pairs': lambda s: s.target.pairs + lag(s.target.pairs, 1),
        }


class RickerSystem(BaseSystem):
    """Ricker's map of a population in discrete generations, `n' = r n exp(-n)`.

    It is chaotic for large `r`.

    The positive fixed point `n = ln r` is stable while `1 < r < e^2`. Beyond `e^2`, about 7.39, it loses stability to a
    two-cycle, and further period doublings lead to chaos near `r = 14.8`. The size stays positive for every `r`. The
    system has the population `population.x`.

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

    def _build_system(self) -> CompositeSystem:
        """Return the components of Ricker's map of a population in discrete generations."""
        return CompositeSystem(
            self.label,
            elements=[PopulationElement('population', x=self.n)],
            interactions=[RickerGrowthInteraction('growth', target='population', r=self.r)],
        )


class PenaeidShrimpSystem(DynamicalSystem):
    """The growth of one penaeid shrimp from the juvenile stage.

    The growth is forced by the food and the water temperature.

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
        >>> PenaeidShrimpSystem().prepare().system_.elements[0].get_value('K')
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
        super().__init__(label=label)
        self.gut = gut
        self.biomass = biomass
        self.ration = ration
        self.respiration = respiration
        self.L_inf = L_inf
        self.K = K

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'shrimp'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'gut': DeclaredVariableSpec(self.gut, StateSpec(domain=RealSet.closed_open(0, oo))),
            'biomass': DeclaredVariableSpec(self.biomass, StateSpec(domain=RealSet.closed_open(0, oo))),
            'gonads': DeclaredVariableSpec(0.0001, StateSpec(domain=RealSet.closed_open(0, oo))),
            'length': DeclaredVariableSpec(0.0, StateSpec(domain=RealSet.closed_open(0, oo))),
            'ration': DeclaredVariableSpec(self.ration, ParameterSpec(domain=RealSet.open(0, oo))),
            'k': DeclaredVariableSpec(0.12, ParameterSpec(domain=RealSet.open(0, oo))),
            'assimilation': DeclaredVariableSpec(0.8, ParameterSpec(domain=RealSet.open(0, oo))),
            'respiration': DeclaredVariableSpec(self.respiration, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'phi': DeclaredVariableSpec(5.0, ParameterSpec(domain=RealSet.open(0, oo))),
            'gsi': DeclaredVariableSpec(0.05, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'egg_weight': DeclaredVariableSpec(2.0e-6, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'L_inf': DeclaredVariableSpec(self.L_inf, ParameterSpec(domain=RealSet.open(0, oo))),
            'K': DeclaredVariableSpec(self.K, ParameterSpec(domain=RealSet.open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the ingestion, assimilation, respiration, spawning, and growth laws."""
        return {
            'gut': lambda s: self._get_ingestion(s) - self._get_assimilated(s),
            'biomass': lambda s: self._get_assimilated(s) - self._get_respired(s) - self._get_spawning(s),
            'gonads': lambda s: s.target.biomass * s.target.gsi,
            'length': lambda s: s.target.K * (s.target.L_inf - s.target.length),
        }

    def _get_food(self, s: SimpleNamespace) -> Expression:
        """Return `food` from the namespace of an equation."""
        return series(s.t, DAYS, FOOD)

    def _get_temperature(self, s: SimpleNamespace) -> Expression:
        """Return `temperature` from the namespace of an equation."""
        return series(s.t, DAYS, TEMPERATURE)

    def _get_max_ration(self, s: SimpleNamespace) -> Expression:
        """Return `max_ration` from the namespace of an equation."""
        return series(s.target.biomass, *ALLOMETRY) * series(self._get_temperature(s), *THERMAL) * s.target.ration

    def _get_ingestion(self, s: SimpleNamespace) -> Expression:
        """Return `ingestion` from the namespace of an equation."""
        return self._get_max_ration(s) * (1 - exp(-s.target.k * self._get_food(s)))

    def _get_assimilated(self, s: SimpleNamespace) -> Expression:
        """Return `assimilated` from the namespace of an equation."""
        return s.target.gut * s.target.assimilation

    def _get_respired(self, s: SimpleNamespace) -> Expression:
        """Return `respired` from the namespace of an equation."""
        return (
            s.target.respiration
            * exp(0.07 * self._get_temperature(s))
            * (s.target.biomass * s.target.phi) ** 0.185
            * s.target.biomass
        )

    def _get_spawning(self, s: SimpleNamespace) -> Expression:
        """Return `spawning` from the namespace of an equation."""
        return 9.3 * s.target.length**2.8 * s.target.egg_weight
