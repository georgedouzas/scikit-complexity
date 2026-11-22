"""Epidemiology systems: sexually transmitted infections along partnerships.

The infections spread along partnerships that overlap in time.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import JumpLaw, RateLaw, TopologyRequestSpec
from ....core.components.spaces import NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.networks import HostElement, SpreadingInteraction

FRACTION = RealSet.closed(0, 1)


class PersonElement(HostElement):
    """Person of a partnership network, drawn by whether it is infected."""

    visual = VisualSpec('sites', {'states': ['i'], 'low': 0, 'high': 1})


class PartnershipEdgeElement(Element):
    """Partnership between two people, hidden from the drawing."""

    visual = VisualSpec('sites', {'states': []})


class PartnershipEpidemicElement(Element):
    """Population of Watts and May, as fractions of susceptible, exposed, infectious, and removed people.

    The infection memory `J` is the infection by current partners who turned infectious after the partnership formed.

    Args:
        label:
            The label of the population.

        S:
            The initial fraction of susceptible people.

        E:
            The initial fraction of exposed people.

        I:
            The initial fraction of infectious people.

        R:
            The initial fraction of people who developed the disease.

        J:
            The initial infection memory of current partnerships.

        values:
            Further values by name, such as the infectious before the start `'I[-1]'`.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import PartnershipEpidemicElement
        >>> PartnershipEpidemicElement('population').S
        0.99
    """

    def __init__(
        self,
        label: str,
        *,
        S: Annotated[float, FRACTION] = 0.99,
        E: Annotated[float, FRACTION] = 0.0,
        I: Annotated[float, FRACTION] = 0.01,
        R: Annotated[float, FRACTION] = 0.0,
        J: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        values: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(label, values=values)
        self.S = S
        self.E = E
        self.I = I
        self.R = R
        self.J = J


class PartnershipInfectionInteraction(Interaction):
    """Infection of Watts and May along lasting partnerships, with a fixed exposed period.

    Susceptible people `S` acquire new partners at the rate `c`, each infecting with the probability `beta` over the
    partnership, and are infected at the rate `pi = c beta (I / A + J)` with the active population `A = S + E + I`. The
    first term is the infection by a new partner who is infectious, and the memory `J` is the infection by a current
    partner who turned infectious after the partnership formed, `dJ/dt = pi(t - T) S(t - T) / A - J / tau`, where
    partnerships last a mean time `tau`. The exposed `E` turn infectious after the fixed time `T`, and the infectious
    `I` develop the disease, which ends their sexual activity, at the rate `nu`, into `R`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the population.

        c:
            The rate of acquiring new partners.

        beta:
            The probability of infection by an infectious partner over a partnership.

        tau:
            The mean duration of a partnership.

        T:
            The duration of the exposed period.

        nu:
            The rate at which the infectious develop the disease.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import PartnershipInfectionInteraction
        >>> PartnershipInfectionInteraction('infection', target='population').tau
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        c: Annotated[float, RealSet.closed_open(0, oo)] = 2.0,
        beta: Annotated[float, FRACTION] = 0.2,
        tau: Annotated[float, RealSet.open(0, oo)] = 1.0,
        T: Annotated[float, RealSet.open(0, oo)] = 0.5,
        nu: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.c = c
        self.beta = beta
        self.tau = tau
        self.T = T
        self.nu = nu

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the infection by new and current partners, the exposed period, and the progression.

        Args:
            context:
                The namespace of the population.
        """
        super()._prepare(context)
        s, t = context.target, context.t
        S, E, I, R, J = s.S, s.E, s.I, s.R, s.J
        active = S + E + I
        infection = context.c * context.beta * (I / active + J) * S
        past = [state(t=t - context.T) for state in (S, E, I, J)]
        past_active = past[0] + past[1] + past[2]
        matured = context.c * context.beta * (past[2] / past_active + past[3]) * past[0]
        self.laws_ = [
            RateLaw(S.diff(t), -infection),
            RateLaw(E.diff(t), infection - matured),
            RateLaw(I.diff(t), matured - context.nu * I),
            RateLaw(R.diff(t), context.nu * I),
            RateLaw(J.diff(t), matured / active - J / context.tau),
        ]


class WattsMayConcurrencySystem(BaseSystem):
    """The SEIR epidemic of Watts and May.

    Partnerships that last longer spread the infection faster.

    A susceptible person is infected at the rate `pi(t) = c beta [I / (S + E + I) + int exp(-(t - t') / tau) pi(t' - T)
    S(t' - T) / (S + E + I)(t') dt']`, by a new partner who is infectious or by a current partner who turned infectious
    during the partnership. The mean number of current partners is `c tau`, so a longer partnership raises the mean
    degree of the momentary network, and the infection rate grows with `tau` at fixed `c` and `beta`. The system has the
    fractions `population.S`, `population.E`, `population.I`, `population.R`, and the infection memory `population.J`.

    Args:
        infected:
            The initial fraction of infectious people.

        c:
            The rate of acquiring new partners.

        beta:
            The probability of infection by an infectious partner over a partnership.

        tau:
            The mean duration of a partnership.

        T:
            The duration of the exposed period.

        nu:
            The rate at which the infectious develop the disease.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import WattsMayConcurrencySystem
        >>> WattsMayConcurrencySystem(tau=3.0).prepare().system_.interactions[0].tau
        3.0
    """

    field = 'biology/epidemiology'
    topics = ('concurrency', 'partnerships', 'seir', 'delay', 'hiv')
    references = ('Watts and May 1992', 'Masuda, Miller, and Holme 2021')
    until = 40.0
    dt = 0.25

    def __init__(
        self,
        infected: float = 0.01,
        *,
        c: float = 2.0,
        beta: float = 0.2,
        tau: float = 1.0,
        T: float = 0.5,
        nu: float = 0.1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.infected = infected
        self.c = c
        self.beta = beta
        self.tau = tau
        self.T = T
        self.nu = nu

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the SEIR epidemic of Watts and May.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        population = PartnershipEpidemicElement(
            'population',
            S=1 - self.infected,
            I=self.infected,
            values={'I[-1]': 0.0, 'J[-1]': 0.0},
        )
        infection = PartnershipInfectionInteraction(
            'infection',
            target='population',
            c=self.c,
            beta=self.beta,
            tau=self.tau,
            T=self.T,
            nu=self.nu,
        )
        self.system_ = System(self.label, elements=[population], interactions=[infection])


class PairFormationSISSystem(BaseSystem):
    """The SIS epidemic on partnerships that dissolve and are replaced at once.

    Every person keeps `k` partners at any time, on a random regular network. A partnership dissolves at the rate
    `sigma`, and the person who keeps it finds a new partner at once, at random. An infectious person infects a
    susceptible partner at the rate `beta` and recovers at the rate `gamma`. By default the infection pressure `k beta`
    and the rate `k sigma` of new partners per person are held fixed at `pressure` and `turnover`, so `k = 1` is serial
    monogamy and a larger `k` is concurrency at the same contact rate. Concurrency lets the infection pass on before the
    partnership that brought it ends, so the epidemic grows faster and reaches more people. The system has the states
    `people.i` of the variable group `people` on the nodes and the variable group `partnerships` on the edges.

    Args:
        n:
            The number of people, even when `k` is odd.

        k:
            The number of concurrent partners of every person.

        beta:
            The infection rate per partnership, or `None` for `pressure / k`.

        gamma:
            The recovery rate.

        sigma:
            The dissolution rate of a partnership, or `None` for `turnover / k`.

        pressure:
            The infection pressure of a person whose partners are all infectious, used when `beta` is `None`.

        turnover:
            The rate of new partners per person, used when `sigma` is `None`.

        infected:
            The initial share of infectious people.

        seed:
            The seed of the network and of the initial infections.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import PairFormationSISSystem
        >>> PairFormationSISSystem(n=20, k=4).prepare().system_.interactions[0].beta
        0.375
    """

    field = 'biology/epidemiology'
    topics = ('concurrency', 'partnerships', 'temporal network', 'sis', 'jumps')
    references = ('Morris and Kretzschmar 1997', 'Masuda, Miller, and Holme 2021')
    until = 40.0
    dt = 0.5

    def __init__(
        self,
        n: int = 400,
        k: int = 2,
        *,
        beta: float | None = None,
        gamma: float = 0.5,
        sigma: float | None = None,
        pressure: float = 1.5,
        turnover: float = 2.0,
        infected: float = 0.05,
        seed: int | None = 1,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n = n
        self.k = k
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.pressure = pressure
        self.turnover = turnover
        self.infected = infected
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the SIS epidemic on partnerships that are replaced at once.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        beta = self.beta
        sigma = self.sigma
        beta = self.pressure / self.k if beta is None else beta
        sigma = self.turnover / self.k if sigma is None else sigma
        space = NetworkSpace(nx.random_regular_graph(self.k, self.n, seed=self.seed))
        initial = (np.random.default_rng(self.seed).random(self.n) < self.infected).astype(float)
        people = PersonElement('people', i=initial, space=space)
        partnerships = PartnershipEdgeElement('partnerships', space=space, on='edges')
        self.system_ = System(
            self.label,
            elements=[people, partnerships],
            interactions=[
                SpreadingInteraction('spreading', target='people', state='i', beta=beta, gamma=self.gamma),
                Interaction(
                    'dissolution',
                    roles={'nodes': 'people'},
                    laws=lambda s: [JumpLaw(sigma, {TopologyRequestSpec('rewire', keep='source'): 1})],
                    along=True,
                ),
            ],
        )
