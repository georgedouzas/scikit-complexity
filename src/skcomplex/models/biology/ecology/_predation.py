"""Ecology models: predation with saturation, parasitism, harvesting, and demographic noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ..._catalog import register_model
from ..common.populations import ExponentialGrowthInteraction, MortalityInteraction, PopulationElement
from ._parasitism import ParasitismInteraction
from ._predation_interaction import PredationInteraction
from ._predator_prey_events import PredatorPreyEventsInteraction
from ._rosenzweig_macarthur import RosenzweigMacArthurInteraction


@register_model(
    field='biology/ecology',
    tags=('predator-prey', 'ode', 'tipping', 'rate-induced tipping', 'slow-fast'),
    references=('Rosenzweig and MacArthur 1963', 'Datseris and Parlitz 2022'),
    defaults={'until': 150.0, 'dt': 0.5, 'solver': ScipyODESolver(rtol=1e-8, atol=1e-300)},
)
def rosenzweig_macarthur(
    prey: float | None = None,
    predators: float | None = None,
    *,
    kappa: float = 0.01,
    eta: float = 0.8,
    phi: float | None = None,
    phi_end: float | None = None,
    rate: float = 0.002,
) -> System:
    """Return fast prey and slow predators whose carrying capacity shrinks at a rate, which tips them when it is fast.

    By default the crowding rises from `eta (1 - eta) / (1 + eta)` to `1 - eta`, the range where the coexistence point
    at `x = 1 / (1 - eta)` is stable, and the run starts there. A slow rise lets the populations track the point, and a
    rise faster than about 0.0004 makes the prey collapse for a while before they return.

    In a collapse the prey fall to about `1e-30`, far below any absolute tolerance, and an error of either sign there
    decides whether they return or, below zero, run into the pole of the feeding at `x = -1 / eta`. The registered
    solver therefore controls the relative error only, with `ScipyODESolver(rtol=1e-8, atol=1e-300)`.

    Args:
        prey:
            The initial prey `x`, or `None` for the coexistence point.

        predators:
            The initial predators `y`, or `None` for the coexistence point.

        kappa:
            The ratio of the time scales of the prey and the predators.

        eta:
            The handling time of the predators.

        phi:
            The initial crowding, or `None` for `eta (1 - eta) / (1 + eta)`.

        phi_end:
            The final crowding, or `None` for `1 - eta`.

        rate:
            The rate at which the crowding rises.

    Returns:
        system:
            The model, with the populations `prey` and `predators`.

    Examples:
        >>> from skcomplex.models.biology.ecology import rosenzweig_macarthur
        >>> round(rosenzweig_macarthur().elements[0].n, 6)
        5.0
    """
    phi = eta * (1 - eta) / (1 + eta) if phi is None else phi
    phi_end = 1 - eta if phi_end is None else phi_end
    prey = 1 / (1 - eta) if prey is None else prey
    predators = (1 - phi * prey) * (1 + eta * prey) if predators is None else predators
    return System(
        elements=[PopulationElement('prey', n=prey), PopulationElement('predators', n=predators)],
        interactions=[
            RosenzweigMacArthurInteraction(
                'predation',
                prey='prey',
                predators='predators',
                kappa=kappa,
                eta=eta,
                phi=phi,
                phi_end=phi_end,
                rate=rate,
            ),
        ],
    )


@register_model(
    field='biology/ecology',
    tags=('host-parasitoid', 'discrete map', 'oscillation'),
    references=('Nicholson and Bailey 1935', 'Argun et al. 2021'),
    defaults={'until': 30, 'dt': 1},
)
def nicholson_bailey(
    hosts: float = 25.0,
    parasitoids: float = 10.0,
    *,
    k: float = 2.0,
    a: float = 0.05,
    c: float = 1.0,
) -> System:
    """Return hosts and parasitoids in discrete generations, whose equilibrium is unstable to growing oscillations.

    The equilibrium is `H = k ln k / ((k - 1) a c)` and `P = ln k / a`.

    Args:
        hosts:
            The initial number of hosts.

        parasitoids:
            The initial number of parasitoids.

        k:
            The reproductive rate of the hosts.

        a:
            The searching efficiency of the parasitoids.

        c:
            The number of viable eggs a parasitoid lays in a host.

    Returns:
        system:
            The model, with the populations `hosts` and `parasitoids`.

    Examples:
        >>> from skcomplex.models.biology.ecology import nicholson_bailey
        >>> nicholson_bailey(k=3.0).interactions[0].k
        3.0
    """
    return System(
        elements=[PopulationElement('hosts', n=hosts), PopulationElement('parasitoids', n=parasitoids)],
        interactions=[ParasitismInteraction('parasitism', host='hosts', parasitoid='parasitoids', k=k, a=a, c=c)],
    )


@register_model(
    field='biology/ecology',
    tags=('predator-prey', 'ode', 'harvesting', 'oscillation'),
    references=('Volterra 1926', 'Trzaska 2023'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def lotka_volterra_with_harvesting(
    prey: float = 16.0,
    predators: float = 8.0,
    *,
    r: float = 1.0,
    a: float = 0.1,
    b: float = 0.05,
    d: float = 0.5,
    prey_harvest: float = 0.4,
    predator_harvest: float = 0.4,
) -> System:
    """Return Lotka-Volterra predation where both species are harvested with a constant effort.

    The populations cycle around `prey = (d + predator_harvest) / b` and `predators = (r - prey_harvest) / a`, which are
    also their averages over a cycle. Harvesting raises the average prey and lowers the average predators, and a prey
    harvest above `r` drives the predators extinct.

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

        prey_harvest:
            The harvesting effort on the prey.

        predator_harvest:
            The harvesting effort on the predators.

    Returns:
        system:
            The model, with the populations `prey` and `predators`.

    Examples:
        >>> from skcomplex.models.biology.ecology import lotka_volterra_with_harvesting
        >>> [interaction.label for interaction in lotka_volterra_with_harvesting().interactions][-2:]
        ['prey_harvest', 'predator_harvest']
    """
    return System(
        elements=[PopulationElement('prey', n=prey), PopulationElement('predators', n=predators)],
        interactions=[
            ExponentialGrowthInteraction('growth', target='prey', r=r),
            PredationInteraction('predation', predator='predators', prey='prey', a=a, b=b),
            MortalityInteraction('death', target='predators', d=d),
            MortalityInteraction('prey_harvest', target='prey', d=prey_harvest),
            MortalityInteraction('predator_harvest', target='predators', d=predator_harvest),
        ],
    )


@register_model(
    field='biology/ecology',
    tags=('predator-prey', 'jump process', 'gillespie', 'paradox of enrichment'),
    references=('Carmichael and Hadžikadić 2019', 'Rosenzweig 1971'),
    defaults={'until': 100.0, 'dt': 0.5},
)
def stochastic_predator_prey(
    prey: float = 200.0,
    predators: float = 200.0,
    *,
    r: float = 1.0,
    K: float = 600.0,
    a: float = 0.005,
    h: float = 0.5,
    e: float = 0.5,
    m: float = 0.3,
    stochastic: bool = True,
) -> System:
    """Return prey and predators born and killed one at a time, with crowding of the prey and saturating predation.

    The deterministic limit has the coexistence point `x = m / (a (e - m h))`. It is stable while
    `K < 2 x + 1 / (a h)`, and enriching the prey beyond that turns it into large cycles, which demographic noise drives
    to extinction.

    Args:
        prey:
            The initial number of prey.

        predators:
            The initial number of predators.

        r:
            The birth rate of the prey.

        K:
            The carrying capacity of the prey.

        a:
            The attack rate.

        h:
            The handling time.

        e:
            The number of predators born per prey eaten.

        m:
            The death rate of the predators.

        stochastic:
            Whether births, deaths, and kills are random events.

    Returns:
        system:
            The model, with the populations `prey` and `predators`.

    Examples:
        >>> from skcomplex.models.biology.ecology import stochastic_predator_prey
        >>> stochastic_predator_prey(K=1000.0).interactions[0].K
        1000.0
    """
    return System(
        elements=[PopulationElement('prey', n=prey), PopulationElement('predators', n=predators)],
        interactions=[
            PredatorPreyEventsInteraction(
                'events',
                prey='prey',
                predators='predators',
                r=r,
                K=K,
                a=a,
                h=h,
                e=e,
                m=m,
                stochastic=stochastic,
            ),
        ],
    )
