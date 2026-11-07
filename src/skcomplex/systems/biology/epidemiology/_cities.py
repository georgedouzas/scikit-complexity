"""Epidemiology systems: epidemics on the contacts of people who visit the same locations of a synthetic city."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import networkx as nx
import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import System
from ..._catalog import register_system
from ._colocation import ColocationInteraction
from ._commuter import CommuterElement


@register_system(
    field='biology/epidemiology',
    tags=('agents', 'contact network', 'bipartite', 'commuting', 'episims'),
    references=('Eubank et al. 2004', 'Raman 2021'),
    defaults={'until': 200},
)
def episims_city(
    people: int = 1000,
    *,
    household: float = 3.0,
    workplace: float = 20.0,
    p: float = 0.05,
    gamma: float = 0.1,
    infected: int = 5,
    closed: bool = False,
    seed: int | None = 0,
) -> System:
    """Return a synthetic city in the spirit of EpiSims, where people infect those they meet at home and at work.

    EpiSims models a city as a bipartite graph of people and locations whose edges are visits labelled with their
    times, and spreads disease between people at the same location. Here every person belongs to a household of mean
    size `household` and a workplace or school of mean size `workplace`, the people sleep at home and work by day,
    and the disease spreads by SIR contacts at every location. The locations form a network whose edges join a home to
    the workplaces of its members, the projection of the bipartite graph onto the locations. Households spread the
    disease within families and workplaces between them, so closing the workplaces, one of the interventions EpiSims
    was built to compare, confines outbreaks to households.

    Args:
        people:
            The number of people.

        household:
            The mean size of a household.

        workplace:
            The mean size of a workplace or school.

        p:
            The probability of transmission per infected contact and half day.

        gamma:
            The probability of recovery per half day.

        infected:
            The initial number of infected people.

        closed:
            Whether the workplaces are closed.

        seed:
            The seed of the households, the workplaces, and the initial infections.

    Returns:
        system:
            The system, with the homes, workplaces, and health states of the agent group `people` on the locations.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import episims_city
        >>> episims_city(100).elements[0].size
        100
    """
    rng = np.random.default_rng(seed)
    homes = rng.integers(0, max(1, round(people / household)), people)
    n_homes = int(homes.max()) + 1
    works = n_homes + rng.integers(0, max(1, round(people / workplace)), people)
    graph = nx.Graph()
    graph.add_nodes_from(range(int(works.max()) + 1))
    graph.add_edges_from(zip(homes.tolist(), works.tolist(), strict=True))
    health = np.zeros(people)
    health[rng.choice(people, size=infected, replace=False)] = 1.0
    group = GroupElement(
        'people',
        element=CommuterElement,
        size=people,
        values={'home': homes.astype(float), 'work': works.astype(float), 'health': health},
        location=homes.tolist(),
    )
    contacts = ColocationInteraction('contacts', people='people', p=p, gamma=gamma, closed=closed)
    return System(elements=[group], interactions=[contacts], space=NetworkSpace(graph), time=DiscreteTime())
