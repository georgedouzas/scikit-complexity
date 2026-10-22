"""Epidemiology models: infections among individuals who walk on grids and travel on networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import networkx as nx
import numpy as np

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import AgentType, LatticeSpace, Law, NetworkSpace, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

SUSCEPTIBLE, EXPOSED, INFECTED, RECOVERED, DEAD = 0.0, 1.0, 2.0, 3.0, 4.0
HEALTHY, SICK, DECEASED = 0.0, 1.0, 2.0


def _wander(agent: AgentView, model: ModelView, probability: float) -> None:
    """Move an agent to a random neighbouring cell or node with a probability."""
    cells = agent.neighbor_cells()
    if cells and model.rng.random() < probability:
        agent.move_to(cells[int(model.rng.integers(len(cells)))])


def _initial_health(size: int, sick: int, rng: np.random.Generator, value: float) -> np.ndarray:
    """Return the health of a group with a number of randomly chosen sick members."""
    health = np.zeros(size)
    health[rng.choice(size, size=sick, replace=False)] = value
    return health


class Person(AgentType):
    """Person of the agent-based SIR model, with incubation, mortality, waning immunity, and lockdown.

    A person walks to a neighbouring cell with probability `d`, or `lockdown_d` during the lockdown, which lasts
    `lockdown_length` steps from `lockdown_start`. An infectious person infects, with probability `beta`, every
    susceptible person in its cell, who incubates the disease for `incubation` steps. An infectious person recovers
    with probability `gamma` or dies with probability `mu`, and a recovered person loses immunity with probability
    `alpha`. Health is 0 susceptible, 1 exposed, 2 infected, 3 recovered, or 4 dead.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell, or `None` for a random cell.

        health:
            The health state.

        clock:
            The steps since infection.

        d:
            The probability of moving at every step.

        beta:
            The probability that an infectious person infects the susceptible persons of its cell.

        gamma:
            The probability of recovering at every step.

        mu:
            The probability of dying at every step of the infection.

        alpha:
            The probability of losing immunity at every step.

        incubation:
            The number of steps before an infected person becomes infectious.

        lockdown_start:
            The step at which the lockdown starts.

        lockdown_length:
            The number of steps of the lockdown, 0 for none.

        lockdown_d:
            The probability of moving during the lockdown.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import Person
        >>> Person.group('people', 100).size
        100
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 4.0)})
    health: float = state(default=SUSCEPTIBLE, domain=Domain.NONNEGATIVE_INTEGER)
    clock: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    d: float = parameter(default=0.8, domain=Domain.NONNEGATIVE)
    beta: float = parameter(default=0.6, domain=Domain.NONNEGATIVE)
    gamma: float = parameter(default=0.01, domain=Domain.NONNEGATIVE)
    mu: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    alpha: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    incubation: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    lockdown_start: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    lockdown_length: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)
    lockdown_d: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def _infect(self, agent: AgentView, model: ModelView) -> None:
        """Infect every susceptible person of the cell with probability `beta`."""
        if model.rng.random() >= agent.beta:
            return
        for other in agent.agents_here(agent.group):
            if other.health == SUSCEPTIBLE:
                other.health = EXPOSED if agent.incubation > 0 else INFECTED
                other.clock = 0.0

    def _progress(self, agent: AgentView, model: ModelView) -> None:
        """Advance the course of the disease of an exposed, infected, or recovered person."""
        if agent.health == EXPOSED:
            agent.clock += 1
            if agent.clock >= agent.incubation:
                agent.health = INFECTED
        elif agent.health == INFECTED:
            draw = model.rng.random()
            if draw < agent.gamma:
                agent.health = RECOVERED
            elif draw < agent.gamma + agent.mu:
                agent.health = DEAD
        elif agent.health == RECOVERED and model.rng.random() < agent.alpha:
            agent.health = SUSCEPTIBLE

    def laws(self, s: Symbols) -> list[Law]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.live)]

    def live(self, agent: AgentView, model: ModelView) -> None:
        """Walk, infect the cell when infectious, and advance the disease.

        Args:
            agent:
                The person.

            model:
                The model.
        """
        if agent.health == DEAD:
            return
        locked = agent.lockdown_start <= model.time < agent.lockdown_start + agent.lockdown_length
        _wander(agent, model, agent.lockdown_d if locked else agent.d)
        if agent.health == INFECTED:
            self._infect(agent, model)
        self._progress(agent, model)


class Walker(AgentType):
    """Random walker that carries a deadly disease, of Charbonneau's epidemic model.

    A living walker moves to a neighbouring cell or node with probability `travel`. A sick walker infects every
    healthy walker of its cell with probability `infection` at every step, and falls dead `lifetime` steps after its
    own infection. Health is 0 healthy, 1 sick, or 2 dead.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial cell or node, or `None` for a random one.

        health:
            The health state.

        remaining:
            The steps a sick walker has left to live.

        lifetime:
            The number of steps a walker lives after its infection.

        infection:
            The probability that a sick walker infects a healthy walker of its cell at every step.

        travel:
            The probability of moving at every step.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import Walker
        >>> Walker.group('walkers', 10).size
        10
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 2.0)})
    health: float = state(default=HEALTHY, domain=Domain.NONNEGATIVE_INTEGER)
    remaining: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    lifetime: float = parameter(default=20.0, domain=Domain.POSITIVE)
    infection: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)
    travel: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.walk)]

    def walk(self, agent: AgentView, model: ModelView) -> None:
        """Walk, and when sick, infect the walkers of the cell and age towards death.

        Args:
            agent:
                The walker.

            model:
                The model.
        """
        if agent.health == DECEASED:
            return
        _wander(agent, model, agent.travel)
        if agent.health != SICK:
            return
        for other in agent.agents_here(agent.group):
            if other.health == HEALTHY and model.rng.random() < agent.infection:
                other.health = SICK
                other.remaining = other.lifetime
        agent.remaining -= 1
        if agent.remaining <= 0:
            agent.health = DECEASED


class Traveller(AgentType):
    """Resident of a city at a network node, who infects fellow residents and travels to neighbouring cities.

    An infected traveller infects every susceptible traveller of its node with probability `tau`, recovers after
    `days` steps, and moves to a random neighbouring node with probability `m`. Health is 0 susceptible, 2 infected,
    or 3 recovered.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial node, or `None` for a random node.

        health:
            The health state.

        clock:
            The steps since infection.

        tau:
            The probability of infecting a susceptible traveller of the same node at every step.

        days:
            The number of steps an infection lasts.

        m:
            The probability of moving to a neighbouring node at every step.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import Traveller
        >>> Traveller.group('travellers', 10).size
        10
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 3.0)})
    health: float = state(default=SUSCEPTIBLE, domain=Domain.NONNEGATIVE_INTEGER)
    clock: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    tau: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    days: float = parameter(default=14.0, domain=Domain.POSITIVE)
    m: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.live)]

    def live(self, agent: AgentView, model: ModelView) -> None:
        """Infect and recover when infected, then travel.

        Args:
            agent:
                The traveller.

            model:
                The model.
        """
        if agent.health == INFECTED:
            for other in agent.agents_here(agent.group):
                if other.health == SUSCEPTIBLE and model.rng.random() < agent.tau:
                    other.health = INFECTED
                    other.clock = 0.0
            agent.clock += 1
            if agent.clock > agent.days:
                agent.health = RECOVERED
        _wander(agent, model, agent.m)


@register_model(
    field='biology/epidemiology',
    tags=('compartments', 'agents', 'grid', 'lockdown', 'mortality'),
    references=('Argun et al. 2021', 'Kermack and McKendrick 1927'),
    defaults={'until': 300},
)
def agent_based_sir(
    width: int = 50,
    height: int = 50,
    *,
    people: int = 500,
    infected: int = 5,
    d: float = 0.8,
    beta: float = 0.6,
    gamma: float = 0.02,
    mu: float = 0.0,
    alpha: float = 0.0,
    incubation: float = 0.0,
    lockdown_start: float = 0.0,
    lockdown_length: float = 0.0,
    lockdown_d: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return people who walk on a grid and infect those in their cell, with optional extensions of the SIR model.

    The basic model infects with probability `beta` and recovers with probability `gamma`. Its final size depends on
    `beta` and `gamma` separately, not only on their ratio, because walking limits the contacts. Mortality `mu`, waning
    immunity `alpha`, an incubation of `incubation` steps, and a lockdown that lowers the probability of moving to
    `lockdown_d` extend it.

    Args:
        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

        people:
            The number of people.

        infected:
            The initial number of infected people.

        d:
            The probability of moving at every step.

        beta:
            The probability that an infectious person infects the susceptible persons of its cell.

        gamma:
            The probability of recovering at every step.

        mu:
            The probability of dying at every step of the infection.

        alpha:
            The probability of losing immunity at every step.

        incubation:
            The number of steps before an infected person becomes infectious.

        lockdown_start:
            The step at which the lockdown starts.

        lockdown_length:
            The number of steps of the lockdown, 0 for none.

        lockdown_d:
            The probability of moving during the lockdown.

        random_state:
            The seed or generator of the initial cells and infections.

    Returns:
        system:
            The model, with the group `people`.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import agent_based_sir
        >>> int(agent_based_sir(people=100, infected=3, random_state=0).elements[0].values['health'].sum())
        6
    """
    rng = np.random.default_rng(random_state)
    cells = rng.integers(0, [width, height], size=(people, 2))
    group = Person.group(
        'people',
        people,
        location=[tuple(int(item) for item in cell) for cell in cells],
        health=_initial_health(people, infected, rng, INFECTED),
        d=d,
        beta=beta,
        gamma=gamma,
        mu=mu,
        alpha=alpha,
        incubation=incubation,
        lockdown_start=lockdown_start,
        lockdown_length=lockdown_length,
        lockdown_d=lockdown_d,
    )
    return System(components=[group], space=LatticeSpace((width, height), boundary='fixed', neighborhood='von_neumann'))


@register_model(
    field='biology/epidemiology',
    tags=('agents', 'grid', 'network', 'random walk', 'percolation'),
    references=('Charbonneau 2017',),
    defaults={'until': 300},
)
def random_walker_epidemics(
    width: int = 64,
    height: int = 64,
    *,
    walkers: int = 1000,
    lifetime: float = 20.0,
    infection: float = 1.0,
    travel: float = 1.0,
    graph: object = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return random walkers among whom one sick walker starts a deadly epidemic, on a grid or on a network.

    Sick walkers infect the walkers they meet in a cell and die `lifetime` steps later, so the epidemic spreads as a
    front like a forest fire and burns out. Above a critical density of walkers it percolates across the grid. On a
    network, such as a small world of cities, walkers travel along edges and long links carry the disease far.

    Args:
        width:
            The number of columns of the grid, whose edges walkers cannot cross.

        height:
            The number of rows of the grid.

        walkers:
            The number of walkers.

        lifetime:
            The number of steps a walker lives after its infection.

        infection:
            The probability that a sick walker infects a healthy walker of its cell at every step.

        travel:
            The probability of moving at every step.

        graph:
            A NetworkX graph whose nodes the walkers occupy instead of the grid, or `None` for the grid.

        random_state:
            The seed or generator of the initial places and the first sick walker.

    Returns:
        system:
            The model, with the group `walkers`.

    Examples:
        >>> from skcomplex.models.biology.epidemiology import random_walker_epidemics
        >>> float(random_walker_epidemics(walkers=50).elements[0].values['health'].sum())
        1.0
    """
    rng = np.random.default_rng(random_state)
    if graph is None:
        cells = rng.integers(0, [width, height], size=(walkers, 2))
        locations: list[object] = [tuple(int(item) for item in cell) for cell in cells]
        space: LatticeSpace | NetworkSpace = LatticeSpace((width, height), boundary='fixed', neighborhood='von_neumann')
    else:
        nodes = list(nx.nodes(graph))
        locations = [nodes[int(pick)] for pick in rng.integers(0, len(nodes), size=walkers)]
        space = NetworkSpace(graph)
    health = _initial_health(walkers, 1, rng, SICK)
    group = Walker.group(
        'walkers',
        walkers,
        location=locations,
        health=health,
        remaining=health * lifetime,
        lifetime=lifetime,
        infection=infection,
        travel=travel,
    )
    return System(components=[group], space=space)


@register_model(
    field='biology/epidemiology',
    tags=('compartments', 'agents', 'network', 'transport'),
    references=('Datseris and Parlitz 2022', 'Thurner et al. 2018'),
    defaults={'until': 150},
)
def network_sir(
    graph: object = None,
    *,
    residents: int = 20,
    tau: float = 0.05,
    days: float = 14.0,
    m: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return SIR epidemics in cities at the nodes of a network, carried between cities by travellers.

    Every node starts with the same number of susceptible residents, and one resident of a random node is infected.

    Args:
        graph:
            The NetworkX graph of the cities, or `None` for a Barabási-Albert graph of 20 nodes.

        residents:
            The initial number of travellers per node.

        tau:
            The probability of infecting a susceptible traveller of the same node at every step.

        days:
            The number of steps an infection lasts.

        m:
            The probability of moving to a neighbouring node at every step.

        random_state:
            The seed or generator of the first infection.

    Returns:
        system:
            The model, with the group `travellers`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.biology.epidemiology import network_sir
        >>> network_sir(nx.cycle_graph(5), residents=3).elements[0].size
        15
    """
    rng = np.random.default_rng(random_state)
    graph = nx.barabasi_albert_graph(20, 2, seed=0) if graph is None else graph
    nodes = list(nx.nodes(graph))
    size = residents * len(nodes)
    group = Traveller.group(
        'travellers',
        size,
        location=[node for node in nodes for _ in range(residents)],
        health=_initial_health(size, 1, rng, INFECTED),
        tau=tau,
        days=days,
        m=m,
    )
    return System(components=[group], space=NetworkSpace(graph))
