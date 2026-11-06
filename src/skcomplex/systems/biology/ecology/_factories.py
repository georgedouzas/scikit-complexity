"""Ecology systems: population growth, predation, and invasion fronts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import BaseInteraction, MapInteraction
from ....core.components.laws.utils import lag
from ....core.components.system import System
from ....core.specs import StateSpec, VisualSpec
from ..._catalog import register_system
from ..common.populations import (
    ExponentialGrowthInteraction,
    LogisticGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
)
from ._predation_interaction import PredationInteraction
from ._ricker_growth import RickerGrowthInteraction
from ._seasonal_logistic_growth import SeasonalLogisticGrowthInteraction
from ._shrimp import ShrimpElement


@register_system(
    field='biology/ecology',
    tags=('population', 'ode'),
    references=('Malthus 1798',),
    defaults={'until': 20.0, 'dt': 0.1},
)
def exponential_growth(n: float = 10.0, r: float = 0.1) -> System:
    """Return a population that grows or declines at a constant per-capita rate.

    Args:
        n:
            The initial size.

        r:
            The per-capita growth rate.

    Returns:
        system:
            The system, with the population `population`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import exponential_growth
        >>> exponential_growth(r=0.2).interactions[0].r
        0.2
    """
    return System(
        elements=[PopulationElement('population', n=n)],
        interactions=[ExponentialGrowthInteraction('growth', target='population', r=r)],
    )


@register_system(
    field='biology/ecology',
    tags=('population', 'ode'),
    references=('Verhulst 1838',),
    defaults={'until': 20.0, 'dt': 0.1},
)
def logistic_growth(n: float = 10.0, r: float = 1.0, K: float = 100.0) -> System:
    """Return a population that grows logistically towards its carrying capacity.

    Args:
        n:
            The initial size.

        r:
            The intrinsic growth rate.

        K:
            The carrying capacity.

    Returns:
        system:
            The system, with the population `population`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import logistic_growth
        >>> logistic_growth(K=500.0).interactions[0].K
        500.0
    """
    return System(
        elements=[PopulationElement('population', n=n)],
        interactions=[LogisticGrowthInteraction('growth', target='population', r=r, K=K)],
    )


@register_system(
    field='biology/ecology',
    tags=('population', 'ode', 'forcing', 'periodic'),
    references=('Coleman 1979', 'Trzaska 2023'),
    defaults={'until': 20.0, 'dt': 0.05},
)
def varying_capacity_logistic(
    n: float = 10.0,
    *,
    r: float = 2.0,
    K: float = 100.0,
    amplitude: float = 0.5,
    period: float = 1.0,
) -> System:
    """Return logistic growth under a seasonal carrying capacity, which settles on a cycle of the season's period.

    Whatever its initial size, the population tends to a unique periodic solution, which lags behind the capacity and
    varies less than it.

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

    Returns:
        system:
            The system, with the population `population`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import varying_capacity_logistic
        >>> varying_capacity_logistic(amplitude=0.2).interactions[0].amplitude
        0.2
    """
    return System(
        elements=[PopulationElement('population', n=n)],
        interactions=[
            SeasonalLogisticGrowthInteraction(
                'growth',
                target='population',
                r=r,
                K=K,
                amplitude=amplitude,
                period=period,
            ),
        ],
    )


@register_system(
    field='biology/ecology',
    tags=('population', 'delay', 'oscillation'),
    references=('Hutchinson 1948', 'Erneux 2009'),
    defaults={'until': 100.0, 'dt': 0.1},
)
def hutchinson(n: float = 50.0, *, r: float = 1.0, K: float = 100.0, tau: float = 2.0) -> System:
    """Return Hutchinson's delayed logistic equation, which oscillates around `K` when `r * tau` exceeds π/2.

    The history before the start is the initial size.

    Args:
        n:
            The initial size, also its history.

        r:
            The per-capita growth rate at low density.

        K:
            The carrying capacity.

        tau:
            The delay of the density feedback.

    Returns:
        system:
            The system, with the population `population`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import hutchinson
        >>> hutchinson(tau=1.0).interactions[0].tau
        1.0
    """
    return System(
        elements=[PopulationElement('population', n=n)],
        interactions=[LogisticGrowthInteraction('growth', target='population', r=r, K=K, tau=tau)],
    )


@register_system(
    field='biology/ecology',
    tags=('predator-prey', 'ode', 'oscillation'),
    references=('Lotka 1925', 'Volterra 1926'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def lotka_volterra(
    prey: float = 40.0,
    predators: float = 9.0,
    *,
    r: float = 1.1,
    a: float = 0.4,
    b: float = 0.1,
    d: float = 0.4,
    K: float | None = None,
) -> System:
    """Return prey and predators whose populations oscillate through predation.

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

    Returns:
        system:
            The system, with the populations `prey` and `predators`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import lotka_volterra
        >>> [element.label for element in lotka_volterra().elements]
        ['prey', 'predators']
    """
    growth: BaseInteraction = (
        ExponentialGrowthInteraction('growth', target='prey', r=r)
        if K is None
        else LogisticGrowthInteraction('growth', target='prey', r=r, K=K)
    )
    return System(
        elements=[PopulationElement('prey', n=prey), PopulationElement('predators', n=predators)],
        interactions=[
            growth,
            PredationInteraction('predation', predator='predators', prey='prey', a=a, b=b),
            MortalityInteraction('death', target='predators', d=d),
        ],
    )


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


@register_system(
    field='biology/ecology',
    tags=('population', 'discrete map', 'second order', 'age structure'),
    references=('Fibonacci 1202',),
    defaults={'until': 20, 'dt': 1},
)
def fibonacci_rabbits(pairs: float = 1.0, previous: float = 0.0) -> System:
    """Return Fibonacci's rabbits, `N_{t+1} = N_t + N_{t-1}`, where every pair older than one month breeds a new pair.

    The previous population counts the pairs old enough to breed, so the recurrence is an age structure of two classes.

    Args:
        pairs:
            The number of pairs this month.

        previous:
            The number of pairs last month.

    Returns:
        system:
            The system, with the population `rabbits.pairs` and last month's `rabbits.pairs_lag1`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import fibonacci_rabbits
        >>> fibonacci_rabbits().elements[0].pairs
        1.0
    """
    breeding = MapInteraction(
        'breeding',
        target='rabbits',
        state='pairs',
        function=lambda s: s.target.pairs + lag(s.target.pairs, 1),
    )
    return System(elements=[RabbitsElement('rabbits', pairs=pairs, pairs_lag1=previous)], interactions=[breeding])


@register_system(
    field='biology/ecology',
    tags=('population', 'discrete map', 'period doubling', 'chaos'),
    references=('Ricker 1954', 'Dilão 2023'),
    defaults={'until': 100, 'dt': 1},
)
def ricker(n: float = 0.5, *, r: float = 2.0) -> System:
    """Return Ricker's map of a population in discrete generations, `n' = r n exp(-n)`, chaotic for large `r`.

    The positive fixed point `n = ln r` is stable while `1 < r < e^2`. Beyond `e^2`, about 7.39, it loses stability to
    a two-cycle, and further period doublings lead to chaos near `r = 14.8`. The size stays positive for every `r`.

    Args:
        n:
            The initial size, in units of the density at which the offspring peak.

        r:
            The growth factor at low density.

    Returns:
        system:
            The system, with the population `population.n`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ricker
        >>> ricker(r=12.0).interactions[0].r
        12.0
    """
    return System(
        elements=[PopulationElement('population', n=n)],
        interactions=[RickerGrowthInteraction('growth', target='population', r=r)],
    )


@register_system(
    field='biology/ecology',
    tags=('individual growth', 'bioenergetics', 'aquaculture', 'stocks and flows', 'ode'),
    references=('Franco et al. 2006', 'Bala et al. 2017'),
    defaults={'until': 60.0, 'dt': 0.5},
)
def penaeid_shrimp(
    gut: float = 0.5,
    biomass: float = 1.0,
    *,
    ration: float = 2.5,
    respiration: float = 0.05,
    L_inf: float = 29.9,
    K: float = 0.013,
) -> System:
    """Return the growth of one penaeid shrimp from the juvenile stage, forced by the food and the water temperature.

    The individual ingests food up to a maximum ration set by its biomass and the water temperature, assimilates its
    gut content, respires, and spawns, while its gonads grow and its carapace length follows von Bertalanffy's curve
    `L(t) = L_inf (1 - exp(-K t))` from zero, which nears its asymptote after six to eight months. The gut content
    settles near `I / A_e`, the ingestion over the assimilation efficiency.

    The coefficients are those of the listing of Bala et al. (2017). With them the respiration of a juvenile at 28 C,
    `respiration exp(0.07 T) (phi B)^0.185`, is about half its biomass per day, more than its maximum ration, so the
    total biomass declines slowly while the gonads and the length grow. The weight gain the source describes needs a
    lower `respiration` coefficient, such as 0.02.

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

    Returns:
        system:
            The system, with the states `shrimp.gut`, `shrimp.biomass`, `shrimp.gonads`, and `shrimp.length`.

    Examples:
        >>> from skcomplex.systems.biology.ecology import penaeid_shrimp
        >>> penaeid_shrimp().elements[0].K
        0.013
    """
    return System(
        elements=[
            ShrimpElement(
                'shrimp',
                gut=gut,
                biomass=biomass,
                ration=ration,
                respiration=respiration,
                L_inf=L_inf,
                K=K,
            ),
        ],
    )
