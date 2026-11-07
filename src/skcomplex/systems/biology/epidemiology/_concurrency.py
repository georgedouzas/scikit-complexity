"""Epidemiology systems: sexually transmitted infections spreading along partnerships that overlap in time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx
import numpy as np

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import TopologyInteraction
from ....core.components.spaces import NetworkSpace
from ....core.components.system import System
from ....core.specs import VisualSpec
from ..._catalog import register_system
from ._host import HostElement
from ._partnership_epidemic import PartnershipEpidemicElement
from ._spreading import SpreadingInteraction


@register_system(
    field='biology/epidemiology',
    tags=('concurrency', 'partnerships', 'seir', 'delay', 'hiv'),
    references=('Watts and May 1992', 'Masuda, Miller, and Holme 2021'),
    defaults={'until': 40.0, 'dt': 0.25},
)
def watts_may_concurrency(
    infected: float = 0.01,
    *,
    c: float = 2.0,
    beta: float = 0.2,
    tau: float = 1.0,
    T: float = 0.5,
    nu: float = 0.1,
) -> System:
    """Return the SEIR epidemic of Watts and May, where partnerships that last longer spread the infection faster.

    A susceptible person is infected at the rate `pi(t) = c beta [I / (S + E + I) + int exp(-(t - t') / tau) pi(t' - T)
    S(t' - T) / (S + E + I)(t') dt']`, by a new partner who is infectious or by a current partner who turned infectious
    during the partnership. The mean number of current partners is `c tau`, so a longer partnership raises the mean
    degree of the momentary network, and the infection rate grows with `tau` at fixed `c` and `beta`.

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

    Returns:
        system:
            The system, with the fractions `population.S`, `population.E`, `population.I`, `population.R`, and the
            infection memory `population.J`.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import watts_may_concurrency
        >>> watts_may_concurrency(tau=3.0).elements[0].tau
        3.0
    """
    population = PartnershipEpidemicElement(
        'population',
        S=1 - infected,
        I=infected,
        c=c,
        beta=beta,
        tau=tau,
        T=T,
        nu=nu,
    )
    return System(elements=[population])


@register_system(
    field='biology/epidemiology',
    tags=('concurrency', 'partnerships', 'temporal network', 'sis', 'jumps'),
    references=('Morris and Kretzschmar 1997', 'Masuda, Miller, and Holme 2021'),
    defaults={'until': 40.0, 'dt': 0.5},
)
def pair_formation_sis(
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
) -> System:
    """Return the SIS epidemic on partnerships that dissolve and are replaced at once, with `k` partners each.

    Every person keeps `k` partners at any time, on a random regular network. A partnership dissolves at the rate
    `sigma`, and the person who keeps it finds a new partner at once, at random. An infectious person infects a
    susceptible partner at the rate `beta` and recovers at the rate `gamma`. By default the infection pressure `k beta`
    and the rate `k sigma` of new partners per person are held fixed at `pressure` and `turnover`, so `k = 1` is serial
    monogamy and a larger `k` is concurrency at the same contact rate. Concurrency lets the infection pass on before
    the partnership that brought it ends, so the epidemic grows faster and reaches more people.

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

    Returns:
        system:
            The system, with the states `people.i` of the variable group `people` on the nodes and the variable group
            `partnerships` on the edges.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import pair_formation_sis
        >>> pair_formation_sis(n=20, k=4).interactions[0].beta
        0.375
    """
    beta = pressure / k if beta is None else beta
    sigma = turnover / k if sigma is None else sigma
    space = NetworkSpace(nx.random_regular_graph(k, n, seed=seed))
    initial = (np.random.default_rng(seed).random(n) < infected).astype(float)
    people = GroupElement(
        'people',
        element=HostElement,
        space=space,
        on='sites',
        variable=True,
        capacity=n,
        values={'i': initial},
        visual=VisualSpec('sites', {'states': ['i'], 'low': 0, 'high': 1}),
    )
    partnerships = GroupElement(
        'partnerships',
        element=Element,
        space=space,
        on='edges',
        variable=True,
        capacity=space.graph.number_of_edges(),
        visual=VisualSpec('sites', {'states': []}),
    )
    return System(
        elements=[people, partnerships],
        interactions=[
            SpreadingInteraction('spreading', target='people', state='i', beta=beta, gamma=gamma),
            TopologyInteraction('dissolution', target='people', action='rewire', rate=sigma, keep='source'),
        ],
    )
