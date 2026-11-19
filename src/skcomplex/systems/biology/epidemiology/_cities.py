"""Epidemiology systems: epidemics among people who share locations of a city.

The epidemics spread on the contacts of people who visit the same locations of a synthetic city.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.spaces import BaseSpace, NetworkSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import AgentView, BaseSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array, IntArray, Role

SUSCEPTIBLE, INFECTED, RECOVERED = 0.0, 1.0, 2.0


class CommuterView(AgentView):
    """View of a commuter, with its states as attributes.

    `CommuterElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    home: float
    work: float
    health: float
    node: int


class CommuterElement(Element):
    """Person of a synthetic city with an SIR health state.

    The person sleeps at home and spends the day at a workplace.

    The home and the workplace are the locations the person visits, the edges of the person in the bipartite graph of
    people and locations. Health is 0 susceptible, 1 infected, or 2 recovered.

    Args:
        label:
            The label of the person or of the group of people.

        home:
            The location of the home.

        work:
            The location of the workplace or school.

        health:
            The health state.

        node:
            The location the person is at, a node of the network of locations.

        size:
            The number of people, or `None` for one.

        space:
            The network of locations, or `None` for none.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import CommuterElement
        >>> CommuterElement('person', home=3.0, work=12.0).work
        12.0
    """

    visual = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 2.0)})

    def __init__(
        self,
        label: str,
        *,
        home: Annotated[float | Array, NN] = 0.0,
        work: Annotated[float | Array, NN] = 0.0,
        health: Annotated[float | Array, NN] = 0.0,
        node: int | IntArray = 0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.home = home
        self.work = work
        self.health = health
        self.node = node

    @classmethod
    def _get_view(cls) -> type[CommuterView]:
        """Return the view of a commuter, `CommuterView`."""
        return CommuterView


class ColocationInteraction(Interaction):
    """Daily movement of people between home and work, and infection at locations.

    A step is half a day: in even steps every person is at home and in odd steps at the workplace, unless workplaces
    are closed, the time-labelled visits of the bipartite graph of people and locations of EpiSims. At every step a
    susceptible person is infected with the probability `1 - (1 - p)^k` by the `k` infected people at the same location,
    and an infected person recovers with the probability `gamma`.

    Args:
        label:
            The label of the interaction.

        people:
            The path of the agent group of people.

        p:
            The probability of transmission per infected contact and step.

        gamma:
            The probability of recovery per step.

        closed:
            Whether the workplaces are closed, so people stay at home.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import ColocationInteraction
        >>> ColocationInteraction('contacts', people='people', p=0.02).p
        0.02
    """

    draws = True

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        people: Role,
        p: float = 0.05,
        gamma: float = 0.1,
        closed: bool = False,
    ) -> None:
        super().__init__(label)
        self.people = people
        self.p = p
        self.gamma = gamma
        self.closed = closed

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rule of a half day.

        Args:
            context:
                The namespace of the symbols the laws can use.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        self.laws_: list[BaseLaw] = [UpdateLaw(context, self.visit)]

    def visit(self, system: SystemView) -> None:
        """Move every person to the location of the half day, then infect and recover.

        Args:
            system:
                The system.
        """
        people = system.get_agents(self.people, CommuterView)
        at_work = round(system.time) % 2 == 1 and not self.closed
        places = np.array([int(person.work if at_work else person.home) for person in people])
        health = np.array([person.health for person in people])
        infected = np.bincount(places[health == INFECTED], minlength=places.max() + 1)[places]
        draws = system.rng.random(len(people))
        infection = (health == SUSCEPTIBLE) & (draws < 1 - (1 - self.p) ** infected)
        recovery = (health == INFECTED) & (draws < self.gamma)
        health = np.where(infection, INFECTED, np.where(recovery, RECOVERED, health))
        for person, place, state in zip(people, places, health, strict=True):
            person.node = int(place)
            person.health = float(state)


class EpisimsCitySystem(BaseSystem):
    """A synthetic city in the spirit of EpiSims.

    People infect those they meet at home and at work.

    EpiSims models a city as a bipartite graph of people and locations whose edges are visits labelled with their times,
    and spreads disease between people at the same location. Here every person belongs to a household of mean size
    `household` and a workplace or school of mean size `workplace`, the people sleep at home and work by day, and the
    disease spreads by SIR contacts at every location. The locations form a network whose edges join a home to the
    workplaces of its members, the projection of the bipartite graph onto the locations. Households spread the disease
    within families and workplaces between them, so closing the workplaces, one of the interventions EpiSims was built
    to compare, confines outbreaks to households. The system has the homes, workplaces, and health states of the agent
    group `people` on the locations.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import EpisimsCitySystem
        >>> EpisimsCitySystem(100).prepare().system_.elements[0].size
        100
    """

    field = 'biology/epidemiology'
    topics = ('agents', 'contact network', 'bipartite', 'commuting', 'episims')
    references = ('Eubank et al. 2004', 'Raman 2021')
    until = 200

    def __init__(
        self,
        people: int = 1000,
        *,
        household: float = 3.0,
        workplace: float = 20.0,
        p: float = 0.05,
        gamma: float = 0.1,
        infected: int = 5,
        closed: bool = False,
        seed: int | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.people = people
        self.household = household
        self.workplace = workplace
        self.p = p
        self.gamma = gamma
        self.infected = infected
        self.closed = closed
        self.seed = seed

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a synthetic city in the spirit of EpiSims.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.seed)
        homes = rng.integers(0, max(1, round(self.people / self.household)), self.people)
        n_homes = int(homes.max()) + 1
        works = n_homes + rng.integers(0, max(1, round(self.people / self.workplace)), self.people)
        graph = nx.Graph()
        graph.add_nodes_from(range(int(works.max()) + 1))
        graph.add_edges_from(zip(homes.tolist(), works.tolist(), strict=True))
        health = np.zeros(self.people)
        health[rng.choice(self.people, size=self.infected, replace=False)] = 1.0
        space = NetworkSpace(graph)
        group = CommuterElement(
            'people',
            home=homes.astype(float),
            work=works.astype(float),
            health=health,
            node=homes,
            size=self.people,
            space=space,
        )
        contacts = ColocationInteraction('contacts', people='people', p=self.p, gamma=self.gamma, closed=self.closed)
        self.system_ = System(self.label, elements=[group], interactions=[contacts], space=space)
