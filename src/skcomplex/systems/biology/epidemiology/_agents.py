"""Epidemiology systems: infections among individuals on grids and networks.

The individuals walk on grids and travel on networks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MovementInteraction, TransitionInteraction
from ....core.components.spaces import LatticeSpace, NetworkSpace
from ....core.components.systems import BaseSystem, System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

SUSCEPTIBLE, EXPOSED, INFECTED, RECOVERED, DEAD = 0.0, 1.0, 2.0, 3.0, 4.0

HEALTHY, SICK, DECEASED = 0.0, 1.0, 2.0


class PersonView(AgentView):
    """View of a person, with its states and parameters as attributes.

    `PersonElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    health: float
    d: float
    beta: float
    gamma: float
    mu: float
    alpha: float
    lockdown_start: float
    lockdown_length: float
    lockdown_d: float


class PersonElement(Element):
    """Person of the agent-based SIR system.

    The person has incubation, mortality, waning immunity, and lockdown.

    A `MovementInteraction` walks a living person to a neighbouring cell with probability `d`, or `lockdown_d` during
    the lockdown, which lasts `lockdown_length` steps from `lockdown_start`. Transitions of the agent form, which
    `AgentBasedSIRSystem` adds, move the health of the person. An infectious person infects, with probability `beta`,
    every susceptible person in its cell, who becomes exposed. An infectious person recovers with probability `gamma`
    or dies with probability `mu`, and a recovered person loses immunity with probability `alpha`. Health is 0
    susceptible, 1 exposed, 2 infected, 3 recovered, or 4 dead.

    The incubation follows the convention of the residence time `after` of `TransitionInteraction`: the step of the
    exposure counts as the first step in the exposed state, so a person exposed in step `n` becomes infectious at the
    end of step `n + incubation - 1`. The clock that the person kept before counted the step of the exposure only when
    the person acted after its infector in that step, so the onset now comes half a step earlier on average.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        health:
            The health state.

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

        lockdown_start:
            The step at which the lockdown starts.

        lockdown_length:
            The number of steps of the lockdown, 0 for none.

        lockdown_d:
            The probability of moving during the lockdown.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.epidemiology import PersonElement
        >>> GroupElement('people', element=PersonElement, size=100).size
        100
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 4.0)}),
        *,
        health: Annotated[float, StateSpec(domain=NN)] = SUSCEPTIBLE,
        d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.6,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        alpha: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lockdown_start: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lockdown_length: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        lockdown_d: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health
        self.d = d
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.lockdown_start = lockdown_start
        self.lockdown_length = lockdown_length
        self.lockdown_d = lockdown_d

    @classmethod
    def _get_view(cls) -> type[PersonView]:
        """Return the view of a person, `PersonView`."""
        return PersonView


class TravellerView(AgentView):
    """View of a traveller, with its states and parameters as attributes.

    `TravellerElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    health: float
    tau: float
    m: float


class TravellerElement(Element):
    """Resident of a city at a network node.

    The resident infects fellow residents and travels to neighbouring cities.

    In its own activation, an infected traveller infects every susceptible traveller of its node with probability
    `tau`, by a `TransitionInteraction` whose members act on their cell, and another transition makes it recover after
    a residence of a number of steps. A `MovementInteraction` moves it to a random neighbouring node with probability
    `m`. Health is 0 susceptible, 2 infected, or 3 recovered.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        health:
            The health state.

        tau:
            The probability of infecting a susceptible traveller of the same node at every step.

        m:
            The probability of moving to a neighbouring node at every step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.epidemiology import TravellerElement
        >>> GroupElement('travellers', element=TravellerElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 3.0)}),
        *,
        health: Annotated[float, StateSpec(domain=NN)] = SUSCEPTIBLE,
        tau: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        m: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health
        self.tau = tau
        self.m = m

    @classmethod
    def _get_view(cls) -> type[TravellerView]:
        """Return the view of a traveller, `TravellerView`."""
        return TravellerView


class WalkerView(AgentView):
    """View of a walker, with its states and parameters as attributes.

    `WalkerElement` names this class with `_get_view`, so its rules read them with checked types.
    """

    health: float
    infection: float
    travel: float


class WalkerElement(Element):
    """Random walker that carries a deadly disease, of Charbonneau's epidemic system.

    In its own activation, a living walker moves to a neighbouring cell or node with probability `travel` by a
    `MovementInteraction`, and a `TransitionInteraction` whose members act on their cell makes a sick walker infect
    every healthy walker of its cell with probability `infection`. Another transition kills a sick walker after a
    residence of a number of steps. Health is 0 healthy, 1 sick, or 2 dead.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        health:
            The health state.

        infection:
            The probability that a sick walker infects a healthy walker of its cell at every step.

        travel:
            The probability of moving at every step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.biology.epidemiology import WalkerElement
        >>> GroupElement('walkers', element=WalkerElement, size=10).size
        10
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'health', 'color_range': (0.0, 2.0)}),
        *,
        health: Annotated[float, StateSpec(domain=NN)] = HEALTHY,
        infection: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        travel: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.health = health
        self.infection = infection
        self.travel = travel

    @classmethod
    def _get_view(cls) -> type[WalkerView]:
        """Return the view of a walker, `WalkerView`."""
        return WalkerView


def _is_due(agent: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return true.

    A member moves as soon as it has held its value for the residence time.
    """
    return True


def _initial_health(size: int, sick: int, rng: np.random.Generator, value: float) -> np.ndarray:
    """Return the health of a group with a number of randomly chosen sick members."""
    health = np.zeros(size)
    health[rng.choice(size, size=sick, replace=False)] = value
    return health


class AgentBasedSIRSystem(BaseSystem):
    """People who walk on a grid and infect those in their cell.

    The basic system infects with probability `beta` and recovers with probability `gamma`. Its final size depends on
    `beta` and `gamma` separately, not only on their ratio, because walking limits the contacts. Mortality `mu`, waning
    immunity `alpha`, an incubation of `incubation` steps, and a lockdown that lowers the probability of moving to
    `lockdown_d` extend it. The system has the group `people`.

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
            The number of steps an infected person stays exposed before it becomes infectious, counting the step of
            the exposure, as the residence time of `TransitionInteraction` counts them.

        lockdown_start:
            The step at which the lockdown starts.

        lockdown_length:
            The number of steps of the lockdown, 0 for none.

        lockdown_d:
            The probability of moving during the lockdown.

        random_state:
            The seed or generator of the initial cells and infections.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import AgentBasedSIRSystem
        >>> system = AgentBasedSIRSystem(people=100, infected=3, random_state=0).prepare().system_
        >>> int(system.elements[0].values['health'].sum())
        6
    """

    field = 'biology/epidemiology'
    topics = ('compartments', 'agents', 'grid', 'lockdown', 'mortality')
    references = ('Argun et al. 2021', 'Kermack and McKendrick 1927')
    until = 300

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.people = people
        self.infected = infected
        self.d = d
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.incubation = incubation
        self.lockdown_start = lockdown_start
        self.lockdown_length = lockdown_length
        self.lockdown_d = lockdown_d
        self.random_state = random_state

    @staticmethod
    def _is_living_person(agent: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether an agent is a living person.

        A living person walks.
        """
        return isinstance(agent, PersonView) and agent.health != DEAD

    @staticmethod
    def _get_person_step(agent: AgentViewProtocol, destination: object, system: SystemViewProtocol) -> float:
        """Return the probability that a person walks.

        The probability is `lockdown_d` during the lockdown and `d` otherwise.
        """
        if not isinstance(agent, PersonView) or not isinstance(system, SystemView):
            return 0.0
        locked = agent.lockdown_start <= system.time < agent.lockdown_start + agent.lockdown_length
        return float(agent.lockdown_d if locked else agent.d)

    @staticmethod
    def _is_spreading(agent: PersonView, system: SystemView) -> bool:
        """Return whether an infectious person spreads in this step.

        The spread is drawn once per step with probability `beta`.

        The infection keeps the draw in its storage for the step, so the person infects every susceptible person of its
        cell or none of them.
        """
        record = system.storage.get(f'spreading.{agent.id}')
        if isinstance(record, tuple) and record[0] == system.time:
            return bool(record[1])
        spreads = bool(system.rng.random() < agent.beta)
        system.storage[f'spreading.{agent.id}'] = (system.time, spreads)
        return spreads

    def _is_person_infected(self, agent: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether an infectious cellmate infects a susceptible person.

        The infectious person shares the cell of the susceptible person, and the infection happens in this step.
        """
        if not isinstance(agent, PersonView) or not isinstance(system, SystemView):
            return False
        return any(
            isinstance(other, PersonView) and other.health == INFECTED and self._is_spreading(other, system)
            for other in agent.agents_here(agent.group)
        )

    @staticmethod
    def _get_person_recovery(agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the probability `gamma` that an infectious person recovers in a step."""
        return float(agent.gamma) if isinstance(agent, PersonView) else 0.0

    @staticmethod
    def _get_person_death(agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the probability `mu / (1 - gamma)` that an infectious person dies.

        It is the probability of death of an infectious person who has not recovered in a step.

        The death follows the recovery in the step, so this conditional probability makes the death `mu` of the step.
        """
        if not isinstance(agent, PersonView) or agent.gamma >= 1:
            return 0.0
        return float(agent.mu / (1 - agent.gamma))

    @staticmethod
    def _get_person_waning(agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the probability `alpha` that a recovered person loses immunity in a step."""
        return float(agent.alpha) if isinstance(agent, PersonView) else 0.0

    @staticmethod
    def _is_incubated(agent: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return true.

        An exposed person becomes infectious as soon as its incubation is over.
        """
        return True

    def _get_person_transitions(self, incubation: float) -> list[TransitionInteraction]:
        """Return the transitions of the people of agent-based SIR.

        They are the infection, waning immunity, recovery, death, and incubation.
        """
        exposed = EXPOSED if incubation > 0 else INFECTED
        transitions = [
            TransitionInteraction(
                'infection',
                target='people',
                state='health',
                schedule='member',
                form='agent',
                source=SUSCEPTIBLE,
                destination=exposed,
                condition=self._is_person_infected,
            ),
            TransitionInteraction(
                'waning',
                target='people',
                state='health',
                schedule='member',
                form='agent',
                source=RECOVERED,
                destination=SUSCEPTIBLE,
                probability=self._get_person_waning,
            ),
            TransitionInteraction(
                'recovery',
                target='people',
                state='health',
                schedule='member',
                form='agent',
                source=INFECTED,
                destination=RECOVERED,
                probability=self._get_person_recovery,
            ),
            TransitionInteraction(
                'death',
                target='people',
                state='health',
                schedule='member',
                form='agent',
                source=INFECTED,
                destination=DEAD,
                probability=self._get_person_death,
            ),
        ]
        if incubation > 0:
            transitions.append(
                TransitionInteraction(
                    'incubation',
                    target='people',
                    state='health',
                    schedule='member',
                    form='agent',
                    source=EXPOSED,
                    destination=INFECTED,
                    condition=self._is_incubated,
                    after=incubation,
                ),
            )
        return transitions

    def _build_system(self) -> System:
        """Return the parts of people who walk on a grid and infect those in their cell."""
        rng = np.random.default_rng(self.random_state)
        cells = rng.integers(0, [self.width, self.height], size=(self.people, 2))
        group = GroupElement(
            'people',
            element=PersonElement,
            size=self.people,
            values={
                'health': _initial_health(self.people, self.infected, rng, INFECTED),
                'd': self.d,
                'beta': self.beta,
                'gamma': self.gamma,
                'mu': self.mu,
                'alpha': self.alpha,
                'lockdown_start': self.lockdown_start,
                'lockdown_length': self.lockdown_length,
                'lockdown_d': self.lockdown_d,
            },
            location=[tuple(int(item) for item in cell) for cell in cells],
        )
        walk = MovementInteraction(
            'walk',
            target='people',
            condition=self._is_living_person,
            accept=self._get_person_step,
        )
        space = LatticeSpace((self.width, self.height), boundary='fixed', neighborhood='von_neumann')
        return System(
            self.label,
            elements=[group],
            interactions=[walk, *self._get_person_transitions(self.incubation)],
            space=space,
        )


class RandomWalkerEpidemicsSystem(BaseSystem):
    """Random walkers among whom one sick walker starts a deadly epidemic.

    The walkers move on a grid or on a network.

    Sick walkers infect the walkers they meet in a cell and die `lifetime` steps later, so the epidemic spreads as a
    front like a forest fire and burns out. Above a critical density of walkers it percolates across the grid. On a
    network, such as a small world of cities, walkers travel along edges and long links carry the disease far. The
    system has the group `walkers`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.epidemiology import RandomWalkerEpidemicsSystem
        >>> float(RandomWalkerEpidemicsSystem(walkers=50).prepare().system_.elements[0].values['health'].sum())
        1.0
    """

    field = 'biology/epidemiology'
    topics = ('agents', 'grid', 'network', 'random walk', 'percolation')
    references = ('Charbonneau 2017',)
    until = 300

    def __init__(
        self,
        width: int = 64,
        height: int = 64,
        *,
        walkers: int = 1000,
        lifetime: float = 20.0,
        infection: float = 1.0,
        travel: float = 1.0,
        graph: object = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.walkers = walkers
        self.lifetime = lifetime
        self.infection = infection
        self.travel = travel
        self.graph = graph
        self.random_state = random_state

    @staticmethod
    def _is_living_walker(agent: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether an agent is a living walker.

        A living walker walks.
        """
        return isinstance(agent, WalkerView) and agent.health != DECEASED

    @staticmethod
    def _get_walker_step(agent: AgentViewProtocol, destination: object, system: SystemViewProtocol) -> float:
        """Return the probability `travel` that a walker takes its step."""
        return float(agent.travel) if isinstance(agent, WalkerView) else 0.0

    @staticmethod
    def _get_walker_infection(agent: AgentViewProtocol, other: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the probability that a walker of the cell infects a healthy walker.

        The probability is `infection` when the walker is sick.
        """
        return float(other.infection) if isinstance(other, WalkerView) and other.health == SICK else 0.0

    def _build_system(self) -> System:
        """Return the parts of random walkers among whom a sick walker starts an epidemic.

        The epidemic is deadly.
        """
        rng = np.random.default_rng(self.random_state)
        if self.graph is None:
            cells = rng.integers(0, [self.width, self.height], size=(self.walkers, 2))
            locations: list[object] = [tuple(int(item) for item in cell) for cell in cells]
            space: LatticeSpace | NetworkSpace = LatticeSpace(
                (self.width, self.height),
                boundary='fixed',
                neighborhood='von_neumann',
            )
        else:
            nodes = list(nx.nodes(self.graph))
            locations = [nodes[int(pick)] for pick in rng.integers(0, len(nodes), size=self.walkers)]
            space = NetworkSpace(self.graph)
        health = _initial_health(self.walkers, 1, rng, SICK)
        group = GroupElement(
            'walkers',
            element=WalkerElement,
            size=self.walkers,
            values={'health': health, 'infection': self.infection, 'travel': self.travel},
            location=locations,
        )
        interactions = [
            MovementInteraction(
                'walk',
                target='walkers',
                condition=self._is_living_walker,
                accept=self._get_walker_step,
                schedule='member',
            ),
            TransitionInteraction(
                'infection',
                target='walkers',
                state='health',
                source=HEALTHY,
                destination=SICK,
                probability=self._get_walker_infection,
                over='cell',
                schedule='member',
                form='agent',
            ),
            TransitionInteraction(
                'death',
                target='walkers',
                state='health',
                source=SICK,
                destination=DECEASED,
                condition=_is_due,
                after=math.ceil(self.lifetime),
                schedule='member',
                form='agent',
            ),
        ]
        return System(self.label, elements=[group], interactions=interactions, space=space)


class NetworkSIRSystem(BaseSystem):
    """SIR epidemics in cities at the nodes of a network.

    Travellers carry the epidemics between cities.

    Every node starts with the same number of susceptible residents, and one resident of a random node is infected. The
    system has the group `travellers`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.systems.biology.epidemiology import NetworkSIRSystem
        >>> NetworkSIRSystem(nx.cycle_graph(5), residents=3).prepare().system_.elements[0].prepare().size_
        15
    """

    field = 'biology/epidemiology'
    topics = ('compartments', 'agents', 'network', 'transport')
    references = ('Datseris and Parlitz 2022', 'Thurner et al. 2018')
    until = 150

    def __init__(
        self,
        graph: object = None,
        *,
        residents: int = 20,
        tau: float = 0.05,
        days: float = 14.0,
        m: float = 0.05,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.graph = graph
        self.residents = residents
        self.tau = tau
        self.days = days
        self.m = m
        self.random_state = random_state

    @staticmethod
    def _get_traveller_step(agent: AgentViewProtocol, destination: object, system: SystemViewProtocol) -> float:
        """Return the probability `m` that a traveller moves to a neighbouring node."""
        return float(agent.m) if isinstance(agent, TravellerView) else 0.0

    @staticmethod
    def _get_traveller_infection(
        agent: AgentViewProtocol,
        other: AgentViewProtocol,
        system: SystemViewProtocol,
    ) -> float:
        """Return the probability that a traveller of the node infects a susceptible one.

        The probability is `tau` when the traveller is infected.
        """
        return float(other.tau) if isinstance(other, TravellerView) and other.health == INFECTED else 0.0

    def _build_system(self) -> System:
        """Return the parts of SIR epidemics in cities at the nodes of a network."""
        graph = self.graph
        rng = np.random.default_rng(self.random_state)
        graph = nx.barabasi_albert_graph(20, 2, seed=0) if graph is None else graph
        nodes = list(nx.nodes(graph))
        size = self.residents * len(nodes)
        group = GroupElement(
            'travellers',
            element=TravellerElement,
            size=size,
            values={'health': _initial_health(size, 1, rng, INFECTED), 'tau': self.tau, 'm': self.m},
            location=[node for node in nodes for _ in range(self.residents)],
        )
        infection = TransitionInteraction(
            'infection',
            target='travellers',
            state='health',
            source=SUSCEPTIBLE,
            destination=INFECTED,
            probability=self._get_traveller_infection,
            over='cell',
            schedule='member',
            form='agent',
        )
        recovery = TransitionInteraction(
            'recovery',
            target='travellers',
            state='health',
            source=INFECTED,
            destination=RECOVERED,
            condition=_is_due,
            after=math.floor(self.days) + 1,
            schedule='member',
            form='agent',
        )
        travel = MovementInteraction('travel', target='travellers', accept=self._get_traveller_step)
        return System(
            self.label,
            elements=[group],
            interactions=[infection, recovery, travel],
            space=NetworkSpace(graph),
        )
