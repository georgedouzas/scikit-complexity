"""Ecology models: predation with saturation, parasitism, harvesting, and demographic noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import exp

from ....core.base import Domain, Expression, parameter, role
from ....core.model import Interaction, Jump, Law, Rate, Symbols, System, Update, where
from ....core.runtime import register_model
from ....core.solvers import ScipyODE
from ._populations import ExponentialGrowth, Mortality, Population, Predation


def _change(target: Expression, rate: Expression, step: int, *, stochastic: bool) -> Law:
    """Return the law that changes a population at a rate, by single individuals in stochastic mode."""
    if stochastic:
        return Jump(rate, {target: step})
    return Rate(target, step * rate)


class RosenzweigMacArthur(Interaction):
    """Rosenzweig-MacArthur predation of fast prey by slow predators, whose prey crowding grows at a constant rate.

    In units of the predators' time scale, `kappa dx/dt = x (1 - phi x) - x y / (1 + eta x)` and
    `dy/dt = x y / (1 + eta x) - y`. The crowding `phi`, the inverse of the prey's carrying capacity, rises at the rate
    `rate` from `phi` to `phi_end` and stays there.

    Args:
        label:
            The label of the interaction.

        prey:
            The path of the prey population `x`.

        predators:
            The path of the predator population `y`.

        kappa:
            The ratio of the time scales of the prey and the predators.

        eta:
            The handling time of the predators.

        phi:
            The initial crowding of the prey.

        phi_end:
            The final crowding of the prey.

        rate:
            The rate at which the crowding rises.

    Examples:
        >>> from skcomplex.models.biology.ecology import RosenzweigMacArthur
        >>> RosenzweigMacArthur('predation', prey='prey', predators='predators', rate=0.01).rate
        0.01
    """

    prey: str = role()
    predators: str = role()
    kappa: float = parameter(default=0.01, domain=Domain.POSITIVE)
    eta: float = parameter(default=0.8, domain=Domain.NONNEGATIVE)
    phi: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    phi_end: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    rate: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rates of the prey and the predators under the ramp of the crowding.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        x, y = s.prey.n, s.predators.n
        crowding = where(s.phi + s.rate * s.t < s.phi_end, s.phi + s.rate * s.t, s.phi_end)
        feeding = x * y / (1 + s.eta * x)
        return [Rate(x, (x * (1 - crowding * x) - feeding) / s.kappa), Rate(y, feeding - y)]


class Parasitism(Interaction):
    """Nicholson-Bailey parasitism of hosts by parasitoids in discrete generations.

    Parasitoids search at random, so a host escapes with probability `exp(-a P)`, giving `H' = k H exp(-a P)` and
    `P' = c H (1 - exp(-a P))`.

    Args:
        label:
            The label of the interaction.

        host:
            The path of the host population `H`.

        parasitoid:
            The path of the parasitoid population `P`.

        k:
            The reproductive rate of the hosts.

        a:
            The searching efficiency of the parasitoids.

        c:
            The number of viable eggs a parasitoid lays in a host.

    Examples:
        >>> from skcomplex.models.biology.ecology import Parasitism
        >>> Parasitism('parasitism', host='moths', parasitoid='wasps', k=2.0).k
        2.0
    """

    host: str = role()
    parasitoid: str = role()
    k: float = parameter(default=2.0, domain=Domain.POSITIVE)
    a: float = parameter(default=0.05, domain=Domain.POSITIVE)
    c: float = parameter(default=1.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the next generations `k H exp(-a P)` and `c H (1 - exp(-a P))`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        hosts, parasitoids = s.host.n, s.parasitoid.n
        escape = exp(-s.a * parasitoids)
        return [Update(hosts, s.k * hosts * escape), Update(parasitoids, s.c * hosts * (1 - escape))]


class PredatorPreyEvents(Interaction):
    """Births, crowding, saturating predation, and deaths of prey and predators as events of single individuals.

    Prey are born at the rate `r x` and die of crowding at the rate `r x^2 / K`. Predators kill prey at the Holling
    type II rate `a x y / (1 + a h x)`, are born at `e` times that rate, and die at the rate `m y`. In deterministic
    mode the rates drive the populations as differential equations.

    Args:
        label:
            The label of the interaction.

        prey:
            The path of the prey population `x`.

        predators:
            The path of the predator population `y`.

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

    Examples:
        >>> from skcomplex.models.biology.ecology import PredatorPreyEvents
        >>> PredatorPreyEvents('events', prey='voles', predators='weasels', K=200.0).K
        200.0
    """

    prey: str = role()
    predators: str = role()
    r: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    K: float = parameter(default=200.0, domain=Domain.POSITIVE)
    a: float = parameter(default=0.02, domain=Domain.NONNEGATIVE)
    h: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    e: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)
    m: float = parameter(default=0.3, domain=Domain.NONNEGATIVE)
    stochastic: bool = True

    def laws(self, s: Symbols) -> list[Law]:
        """Return the five channels of births and deaths.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        x, y = s.prey.n, s.predators.n
        kills = s.a * x * y / (1 + s.a * s.h * x)
        stochastic = self.stochastic
        return [
            _change(x, s.r * x, 1, stochastic=stochastic),
            _change(x, s.r * x**2 / s.K, -1, stochastic=stochastic),
            _change(x, kills, -1, stochastic=stochastic),
            _change(y, s.e * kills, 1, stochastic=stochastic),
            _change(y, s.m * y, -1, stochastic=stochastic),
        ]


@register_model(
    field='biology/ecology',
    tags=('predator-prey', 'ode', 'tipping', 'rate-induced tipping', 'slow-fast'),
    references=('Rosenzweig and MacArthur 1963', 'Datseris and Parlitz 2022'),
    defaults={'until': 150.0, 'dt': 0.5, 'solver': ScipyODE(rtol=1e-8, atol=1e-300)},
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
    solver therefore controls the relative error only, with `ScipyODE(rtol=1e-8, atol=1e-300)`.

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
        elements=[Population('prey', n=prey), Population('predators', n=predators)],
        interactions=[
            RosenzweigMacArthur(
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
        elements=[Population('hosts', n=hosts), Population('parasitoids', n=parasitoids)],
        interactions=[Parasitism('parasitism', host='hosts', parasitoid='parasitoids', k=k, a=a, c=c)],
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
        elements=[Population('prey', n=prey), Population('predators', n=predators)],
        interactions=[
            ExponentialGrowth('growth', target='prey', r=r),
            Predation('predation', predator='predators', prey='prey', a=a, b=b),
            Mortality('death', target='predators', d=d),
            Mortality('prey_harvest', target='prey', d=prey_harvest),
            Mortality('predator_harvest', target='predators', d=predator_harvest),
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
        elements=[Population('prey', n=prey), Population('predators', n=predators)],
        interactions=[
            PredatorPreyEvents(
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
