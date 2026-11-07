"""Ecology systems: predation with saturation, parasitism, harvesting, and demographic noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, exp, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction, JumpInteraction, RateInteraction
from ....core.components.laws import BaseLaw, JumpLaw, RateLaw, UpdateLaw
from ....core.components.laws.utils import clip
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import ParameterSpec, RoleSpec
from ..common.populations import (
    ExponentialGrowthInteraction,
    MortalityInteraction,
    PopulationElement,
    PredationInteraction,
)


class ParasitismInteraction(Interaction):
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
        >>> from skcomplex.systems.biology.ecology import ParasitismInteraction
        >>> ParasitismInteraction('parasitism', host='moths', parasitoid='wasps', k=2.0).k
        2.0
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        host: Annotated[str, RoleSpec()],
        parasitoid: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.host = host
        self.parasitoid = parasitoid
        self.k = k
        self.a = a
        self.c = c

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
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
        return [UpdateLaw(hosts, s.k * hosts * escape), UpdateLaw(parasitoids, s.c * hosts * (1 - escape))]


class PredatorPreyEventsInteraction(Interaction):
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
        >>> from skcomplex.systems.biology.ecology import PredatorPreyEventsInteraction
        >>> PredatorPreyEventsInteraction('events', prey='voles', predators='weasels', K=200.0).K
        200.0
    """

    law_classes = (JumpLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        prey: Annotated[str, RoleSpec()],
        predators: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        K: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 200.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.02,
        h: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        e: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        stochastic: bool = True,
    ) -> None:
        super().__init__(label)
        self.prey = prey
        self.predators = predators
        self.r = r
        self.K = K
        self.a = a
        self.h = h
        self.e = e
        self.m = m
        self.stochastic = stochastic

    def _get_change(self, member: SimpleNamespace, step: int, propensity: Expression) -> Sequence[BaseLaw]:
        """Return the laws that change the population of a member by a step at a propensity.

        In stochastic mode the change is a jump of one individual, and otherwise a rate of the step times the
        propensity.
        """
        change: Interaction = (
            JumpInteraction(self.label, target='member', changes={'n': step}, rate=lambda _: propensity)
            if self.stochastic
            else RateInteraction(self.label, target='member', state='n', rate=lambda _: step * propensity)
        )
        return change.prepare().run(SimpleNamespace(target=member)).laws_

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the five channels of births and deaths.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        prey, predators = s.prey, s.predators
        x, y = prey.n, predators.n
        kills = s.a * x * y / (1 + s.a * s.h * x)
        return [
            *self._get_change(prey, 1, s.r * x),
            *self._get_change(prey, -1, s.r * x**2 / s.K),
            *self._get_change(prey, -1, kills),
            *self._get_change(predators, 1, s.e * kills),
            *self._get_change(predators, -1, s.m * y),
        ]


class RosenzweigMacArthurInteraction(Interaction):
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
        >>> from skcomplex.systems.biology.ecology import RosenzweigMacArthurInteraction
        >>> RosenzweigMacArthurInteraction('predation', prey='prey', predators='predators', rate=0.01).rate
        0.01
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        prey: Annotated[str, RoleSpec()],
        predators: Annotated[str, RoleSpec()],
        kappa: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.01,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        phi: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        phi_end: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label)
        self.prey = prey
        self.predators = predators
        self.kappa = kappa
        self.eta = eta
        self.phi = phi
        self.phi_end = phi_end
        self.rate = rate

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of the prey and the predators under the ramp of the crowding.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        x, y = s.prey.n, s.predators.n
        crowding = clip(s.phi + s.rate * s.t, high=s.phi_end)
        feeding = x * y / (1 + s.eta * x)
        return [RateLaw(x, (x * (1 - crowding * x) - feeding) / s.kappa), RateLaw(y, feeding - y)]


class RosenzweigMacarthurSystem(BaseSystem):
    """Fast prey and slow predators whose carrying capacity shrinks at a rate, which tips them when it is fast.

    By default the crowding rises from `eta (1 - eta) / (1 + eta)` to `1 - eta`, the range where the coexistence point
    at `x = 1 / (1 - eta)` is stable, and the run starts there. A slow rise lets the populations track the point, and a
    rise faster than about 0.0004 makes the prey collapse for a while before they return.

    In a collapse the prey fall to about `1e-30`, far below any absolute tolerance, and an error of either sign there
    decides whether they return or, below zero, run into the pole of the feeding at `x = -1 / eta`. The registered
    solver therefore controls the relative error only, with `ScipyODESolver(rtol=1e-8, atol=1e-300)`. The system has the
    populations `prey` and `predators`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import RosenzweigMacarthurSystem
        >>> round(RosenzweigMacarthurSystem().prepare().system_.elements[0].n, 6)
        5.0
    """

    field = 'biology/ecology'
    topics = ('predator-prey', 'ode', 'tipping', 'rate-induced tipping', 'slow-fast')
    references = ('Rosenzweig and MacArthur 1963', 'Datseris and Parlitz 2022')
    until = 150.0
    dt = 0.5
    solver = ScipyODESolver(rtol=1e-8, atol=1e-300)

    def __init__(
        self,
        prey: float | None = None,
        predators: float | None = None,
        *,
        kappa: float = 0.01,
        eta: float = 0.8,
        phi: float | None = None,
        phi_end: float | None = None,
        rate: float = 0.002,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.prey = prey
        self.predators = predators
        self.kappa = kappa
        self.eta = eta
        self.phi = phi
        self.phi_end = phi_end
        self.rate = rate

    def _build_system(self) -> System:
        """Return the parts of fast prey and slow predators whose carrying capacity shrinks at a rate."""
        prey = self.prey
        predators = self.predators
        phi = self.phi
        phi_end = self.phi_end
        phi = self.eta * (1 - self.eta) / (1 + self.eta) if phi is None else phi
        phi_end = 1 - self.eta if phi_end is None else phi_end
        prey = 1 / (1 - self.eta) if prey is None else prey
        predators = (1 - phi * prey) * (1 + self.eta * prey) if predators is None else predators
        return System(
            self.label,
            elements=[PopulationElement('prey', n=prey), PopulationElement('predators', n=predators)],
            interactions=[
                RosenzweigMacArthurInteraction(
                    'predation',
                    prey='prey',
                    predators='predators',
                    kappa=self.kappa,
                    eta=self.eta,
                    phi=phi,
                    phi_end=phi_end,
                    rate=self.rate,
                ),
            ],
        )


class NicholsonBaileySystem(BaseSystem):
    """Hosts and parasitoids in discrete generations, whose equilibrium is unstable to growing oscillations.

    The equilibrium is `H = k ln k / ((k - 1) a c)` and `P = ln k / a`. The system has the populations `hosts` and
    `parasitoids`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import NicholsonBaileySystem
        >>> NicholsonBaileySystem(k=3.0).prepare().system_.interactions[0].k
        3.0
    """

    field = 'biology/ecology'
    topics = ('host-parasitoid', 'discrete map', 'oscillation')
    references = ('Nicholson and Bailey 1935', 'Argun et al. 2021')
    until = 30
    dt = 1

    def __init__(
        self,
        hosts: float = 25.0,
        parasitoids: float = 10.0,
        *,
        k: float = 2.0,
        a: float = 0.05,
        c: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.hosts = hosts
        self.parasitoids = parasitoids
        self.k = k
        self.a = a
        self.c = c

    def _build_system(self) -> System:
        """Return the parts of hosts and parasitoids in discrete generations."""
        return System(
            self.label,
            elements=[PopulationElement('hosts', n=self.hosts), PopulationElement('parasitoids', n=self.parasitoids)],
            interactions=[
                ParasitismInteraction(
                    'parasitism',
                    host='hosts',
                    parasitoid='parasitoids',
                    k=self.k,
                    a=self.a,
                    c=self.c,
                ),
            ],
        )


class LotkaVolterraWithHarvestingSystem(BaseSystem):
    """Lotka-Volterra predation where both species are harvested with a constant effort.

    The populations cycle around `prey = (d + predator_harvest) / b` and `predators = (r - prey_harvest) / a`, which are
    also their averages over a cycle. Harvesting raises the average prey and lowers the average predators, and a prey
    harvest above `r` drives the predators extinct. The system has the populations `prey` and `predators`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import LotkaVolterraWithHarvestingSystem
        >>> system = LotkaVolterraWithHarvestingSystem().prepare().system_
        >>> [interaction.label for interaction in system.interactions][-2:]
        ['prey_harvest', 'predator_harvest']
    """

    field = 'biology/ecology'
    topics = ('predator-prey', 'ode', 'harvesting', 'oscillation')
    references = ('Volterra 1926', 'Trzaska 2023')
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        prey: float = 16.0,
        predators: float = 8.0,
        *,
        r: float = 1.0,
        a: float = 0.1,
        b: float = 0.05,
        d: float = 0.5,
        prey_harvest: float = 0.4,
        predator_harvest: float = 0.4,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.prey = prey
        self.predators = predators
        self.r = r
        self.a = a
        self.b = b
        self.d = d
        self.prey_harvest = prey_harvest
        self.predator_harvest = predator_harvest

    def _build_system(self) -> System:
        """Return the parts of Lotka-Volterra predation where both species are harvested with a constant effort."""
        return System(
            self.label,
            elements=[PopulationElement('prey', n=self.prey), PopulationElement('predators', n=self.predators)],
            interactions=[
                ExponentialGrowthInteraction('growth', target='prey', r=self.r),
                PredationInteraction('predation', predator='predators', prey='prey', a=self.a, b=self.b),
                MortalityInteraction('death', target='predators', d=self.d),
                MortalityInteraction('prey_harvest', target='prey', d=self.prey_harvest),
                MortalityInteraction('predator_harvest', target='predators', d=self.predator_harvest),
            ],
        )


class StochasticPredatorPreySystem(BaseSystem):
    """Prey and predators born and killed one at a time, with crowding of the prey and saturating predation.

    The deterministic limit has the coexistence point `x = m / (a (e - m h))`. It is stable while `K < 2 x + 1 / (a h)`,
    and enriching the prey beyond that turns it into large cycles, which demographic noise drives to extinction. The
    system has the populations `prey` and `predators`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.ecology import StochasticPredatorPreySystem
        >>> StochasticPredatorPreySystem(K=1000.0).prepare().system_.interactions[0].K
        1000.0
    """

    field = 'biology/ecology'
    topics = ('predator-prey', 'jump process', 'gillespie', 'paradox of enrichment')
    references = ('Carmichael and Hadžikadić 2019', 'Rosenzweig 1971')
    until = 100.0
    dt = 0.5

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.prey = prey
        self.predators = predators
        self.r = r
        self.K = K
        self.a = a
        self.h = h
        self.e = e
        self.m = m
        self.stochastic = stochastic

    def _build_system(self) -> System:
        """Return the parts of prey and predators born and killed one at a time."""
        return System(
            self.label,
            elements=[PopulationElement('prey', n=self.prey), PopulationElement('predators', n=self.predators)],
            interactions=[
                PredatorPreyEventsInteraction(
                    'events',
                    prey='prey',
                    predators='predators',
                    r=self.r,
                    K=self.K,
                    a=self.a,
                    h=self.h,
                    e=self.e,
                    m=self.m,
                    stochastic=self.stochastic,
                ),
            ],
        )
