"""Epidemiology models: infections among individuals who walk on grids and travel on networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import MovementInteraction
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import LatticeSpace, NetworkSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._person_view import PersonView
from ._traveller_view import TravellerView
from ._walker_view import WalkerView

SUSCEPTIBLE, EXPOSED, INFECTED, RECOVERED, DEAD = 0.0, 1.0, 2.0, 3.0, 4.0
HEALTHY, SICK, DECEASED = 0.0, 1.0, 2.0


def _is_living_person(agent: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether an agent is a living person, so it walks."""
    return isinstance(agent, PersonView) and agent.health != DEAD


def _get_person_step(agent: AgentViewProtocol, destination: object, model: ModelViewProtocol) -> float:
    """Return the probability that a person walks, `lockdown_d` during the lockdown and `d` otherwise."""
    if not isinstance(agent, PersonView) or not isinstance(model, ModelView):
        return 0.0
    locked = agent.lockdown_start <= model.time < agent.lockdown_start + agent.lockdown_length
    return float(agent.lockdown_d if locked else agent.d)


def _get_traveller_step(agent: AgentViewProtocol, destination: object, model: ModelViewProtocol) -> float:
    """Return the probability `m` that a traveller moves to a neighbouring node."""
    return float(agent.m) if isinstance(agent, TravellerView) else 0.0


def _initial_health(size: int, sick: int, rng: np.random.Generator, value: float) -> np.ndarray:
    """Return the health of a group with a number of randomly chosen sick members."""
    health = np.zeros(size)
    health[rng.choice(size, size=sick, replace=False)] = value
    return health


class PersonElement(Element):
    """Person of the agent-based SIR model, with incubation, mortality, waning immunity, and lockdown.

    A `MovementInteraction` walks a living person to a neighbouring cell with probability `d`, or `lockdown_d` during
    the lockdown, which lasts `lockdown_length` steps from `lockdown_start`. An infectious person infects, with
    probability `beta`, every susceptible person in its cell, who incubates the disease for `incubation` steps. An
    infectious person recovers with probability `gamma` or dies with probability `mu`, and a recovered person loses
    immunity with probability `alpha`. Health is 0 susceptible, 1 exposed, 2 infected, 3 recovered, or 4 dead.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.biology.epidemiology import PersonElement
        >>> GroupElement('people', element=PersonElement, size=100).size
        100
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 4.0)}),
        *,
        health: Annotated[float, StateSpec(domain=NN)] = SUSCEPTIBLE,
        clock: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        incubation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lockdown_start: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lockdown_length: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lockdown_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health
        self.clock = clock
        self.d = d
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.incubation = incubation
        self.lockdown_start = lockdown_start
        self.lockdown_length = lockdown_length
        self.lockdown_d = lockdown_d

    @classmethod
    def _get_view(cls) -> type[PersonView]:
        """Return the view of a person, `PersonView`."""
        return PersonView

    def _infect(self, agent: PersonView, model: ModelView) -> None:
        """Infect every susceptible person of the cell with probability `beta`."""
        if model.rng.random() >= agent.beta:
            return
        for other in agent.agents_here(agent.group):
            if isinstance(other, PersonView) and other.health == SUSCEPTIBLE:
                other.health = EXPOSED if agent.incubation > 0 else INFECTED
                other.clock = 0.0

    def _progress(self, agent: PersonView, model: ModelView) -> None:
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.live)]

    def live(self, agent: PersonView, model: ModelView) -> None:
        """Infect the cell when infectious, and advance the disease.

        Args:
            agent:
                The person.

            model:
                The model.
        """
        if agent.health == INFECTED:
            self._infect(agent, model)
        self._progress(agent, model)


class WalkerElement(Element):
    """Random walker that carries a deadly disease, of Charbonneau's epidemic model.

    A living walker moves to a neighbouring cell or node with probability `travel`. A sick walker infects every
    healthy walker of its cell with probability `infection` at every step, and falls dead `lifetime` steps after its
    own infection. Health is 0 healthy, 1 sick, or 2 dead.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.biology.epidemiology import WalkerElement
        >>> GroupElement('walkers', element=WalkerElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 2.0)}),
        *,
        health: Annotated[float, StateSpec(domain=NN)] = HEALTHY,
        remaining: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lifetime: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        infection: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        travel: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health
        self.remaining = remaining
        self.lifetime = lifetime
        self.infection = infection
        self.travel = travel

    @classmethod
    def _get_view(cls) -> type[WalkerView]:
        """Return the view of a walker, `WalkerView`."""
        return WalkerView

    def _step(self, agent: WalkerView, model: ModelView) -> None:
        """Move a walker to a random neighbouring cell or node with the probability `travel`."""
        cells = agent.neighbor_cells()
        if cells and model.rng.random() < agent.travel:
            agent.move_to(cells[int(model.rng.integers(len(cells)))])

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.walk)]

    def walk(self, agent: WalkerView, model: ModelView) -> None:
        """Walk, and when sick, infect the walkers of the cell and age towards death.

        The walk stays in the rule of the walker, before its infections, since a walker infected earlier in the same
        step walks on and infects the cell it reaches, which spreads the epidemic several cells in one step.

        Args:
            agent:
                The walker.

            model:
                The model.
        """
        if agent.health == DECEASED:
            return
        self._step(agent, model)
        if agent.health != SICK:
            return
        for other in agent.agents_here(agent.group):
            if isinstance(other, WalkerView) and other.health == HEALTHY and model.rng.random() < agent.infection:
                other.health = SICK
                other.remaining = other.lifetime
        agent.remaining -= 1
        if agent.remaining <= 0:
            agent.health = DECEASED


class TravellerElement(Element):
    """Resident of a city at a network node, who infects fellow residents and travels to neighbouring cities.

    An infected traveller infects every susceptible traveller of its node with probability `tau`, recovers after
    `days` steps, and a `MovementInteraction` moves it to a random neighbouring node with probability `m`. Health is 0
    susceptible, 2 infected, or 3 recovered.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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
        >>> from skcomplex.models.biology.epidemiology import TravellerElement
        >>> GroupElement('travellers', element=TravellerElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 3.0)}),
        *,
        health: Annotated[float, StateSpec(domain=NN)] = SUSCEPTIBLE,
        clock: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        days: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 14.0,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health
        self.clock = clock
        self.tau = tau
        self.days = days
        self.m = m

    @classmethod
    def _get_view(cls) -> type[TravellerView]:
        """Return the view of a traveller, `TravellerView`."""
        return TravellerView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the life rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.live)]

    def live(self, agent: TravellerView, model: ModelView) -> None:
        """Infect and recover when infected.

        Args:
            agent:
                The traveller.

            model:
                The model.
        """
        if agent.health == INFECTED:
            for other in agent.agents_here(agent.group):
                if isinstance(other, TravellerView) and other.health == SUSCEPTIBLE and model.rng.random() < agent.tau:
                    other.health = INFECTED
                    other.clock = 0.0
            agent.clock += 1
            if agent.clock > agent.days:
                agent.health = RECOVERED


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
    group = GroupElement(
        'people',
        element=PersonElement,
        size=people,
        values={
            'health': _initial_health(people, infected, rng, INFECTED),
            'd': d,
            'beta': beta,
            'gamma': gamma,
            'mu': mu,
            'alpha': alpha,
            'incubation': incubation,
            'lockdown_start': lockdown_start,
            'lockdown_length': lockdown_length,
            'lockdown_d': lockdown_d,
        },
        location=[tuple(int(item) for item in cell) for cell in cells],
    )
    walk = MovementInteraction('walk', target='people', condition=_is_living_person, accept=_get_person_step)
    space = LatticeSpace((width, height), boundary='fixed', neighborhood='von_neumann')
    return System(elements=[group], interactions=[walk], space=space)


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
    group = GroupElement(
        'walkers',
        element=WalkerElement,
        size=walkers,
        values={
            'health': health,
            'remaining': health * lifetime,
            'lifetime': lifetime,
            'infection': infection,
            'travel': travel,
        },
        location=locations,
    )
    return System(elements=[group], space=space)


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
        >>> network_sir(nx.cycle_graph(5), residents=3).elements[0].check().prepare().size_
        15
    """
    rng = np.random.default_rng(random_state)
    graph = nx.barabasi_albert_graph(20, 2, seed=0) if graph is None else graph
    nodes = list(nx.nodes(graph))
    size = residents * len(nodes)
    group = GroupElement(
        'travellers',
        element=TravellerElement,
        size=size,
        values={'health': _initial_health(size, 1, rng, INFECTED), 'tau': tau, 'days': days, 'm': m},
        location=[node for node in nodes for _ in range(residents)],
    )
    travel = MovementInteraction('travel', target='travellers', accept=_get_traveller_step)
    return System(elements=[group], interactions=[travel], space=NetworkSpace(graph))
