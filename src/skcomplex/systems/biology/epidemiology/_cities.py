"""Epidemiology systems: epidemics on the contacts of people who visit the same locations of a synthetic city."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.spaces import DiscreteTime, NetworkSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import RoleSpec, StateSpec, VisualSpec

SUSCEPTIBLE, INFECTED, RECOVERED = 0.0, 1.0, 2.0


class CommuterView(AgentView):
    """View of a commuter, with its states as attributes.

    `CommuterElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    home: float
    work: float
    health: float


class CommuterElement(Element):
    """Person of a synthetic city who sleeps at home and spends the day at a workplace, with an SIR health state.

    The home and the workplace are the locations the person visits, the edges of the person in the bipartite graph of
    people and locations. Health is 0 susceptible, 1 infected, or 2 recovered.

    Args:
        label:
            The label of the person or of the group of people.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        home:
            The location of the home.

        work:
            The location of the workplace or school.

        health:
            The health state.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import CommuterElement
        >>> CommuterElement('person', home=3.0, work=12.0).work
        12.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 2.0)}),
        *,
        home: Annotated[float, StateSpec(domain=NN)] = 0.0,
        work: Annotated[float, StateSpec(domain=NN)] = 0.0,
        health: Annotated[float, StateSpec(domain=NN)] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.home = home
        self.work = work
        self.health = health

    @classmethod
    def _get_view(cls) -> type[CommuterView]:
        """Return the view of a commuter, `CommuterView`."""
        return CommuterView


class ColocationInteraction(Interaction):
    """Daily movement of people between their homes and workplaces, and infection between people at one location.

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

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        people: Annotated[str, RoleSpec()],
        p: float = 0.05,
        gamma: float = 0.1,
        closed: bool = False,
    ) -> None:
        super().__init__(label)
        self.people = people
        self.p = p
        self.gamma = gamma
        self.closed = closed

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of a half day.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.visit)]

    def visit(self, system: SystemView) -> None:
        """Move every person to the location of the half day, then infect and recover.

        Args:
            system:
                The system.
        """
        people = system.agents(self.people, CommuterView)
        at_work = round(system.time) % 2 == 1 and not self.closed
        places = np.array([int(person.work if at_work else person.home) for person in people])
        health = np.array([person.health for person in people])
        infected = np.bincount(places[health == INFECTED], minlength=places.max() + 1)[places]
        draws = system.rng.random(len(people))
        infection = (health == SUSCEPTIBLE) & (draws < 1 - (1 - self.p) ** infected)
        recovery = (health == INFECTED) & (draws < self.gamma)
        health = np.where(infection, INFECTED, np.where(recovery, RECOVERED, health))
        for person, place, state in zip(people, places, health, strict=True):
            person.move_to(int(place))
            person.health = float(state)


class EpisimsCitySystem(BaseSystem):
    """A synthetic city in the spirit of EpiSims, where people infect those they meet at home and at work.

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
    ) -> None:
        super().__init__(label)
        self.people = people
        self.household = household
        self.workplace = workplace
        self.p = p
        self.gamma = gamma
        self.infected = infected
        self.closed = closed
        self.seed = seed

    def _build_system(self) -> System:
        """Return the parts of a synthetic city in the spirit of EpiSims."""
        rng = np.random.default_rng(self.seed)
        homes = rng.integers(0, max(1, round(self.people / self.household)), self.people)
        n_homes = int(homes.max()) + 1
        works = n_homes + rng.integers(0, max(1, round(self.people / self.workplace)), self.people)
        graph = nx.Graph()
        graph.add_nodes_from(range(int(works.max()) + 1))
        graph.add_edges_from(zip(homes.tolist(), works.tolist(), strict=True))
        health = np.zeros(self.people)
        health[rng.choice(self.people, size=self.infected, replace=False)] = 1.0
        group = GroupElement(
            'people',
            element=CommuterElement,
            size=self.people,
            values={'home': homes.astype(float), 'work': works.astype(float), 'health': health},
            location=homes.tolist(),
        )
        contacts = ColocationInteraction('contacts', people='people', p=self.p, gamma=self.gamma, closed=self.closed)
        return System(
            self.label,
            elements=[group],
            interactions=[contacts],
            space=NetworkSpace(graph),
            time=DiscreteTime(),
        )
