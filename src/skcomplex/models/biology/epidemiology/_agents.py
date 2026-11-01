"""Epidemiology models: infections among individuals who walk on grids and travel on networks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import networkx as nx
import numpy as np

from ....core.elements import GroupElement
from ....core.interactions import MovementInteraction, TransitionInteraction
from ....core.spaces import LatticeSpace, NetworkSpace
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....engine.solvers import ModelView
from ..._catalog import register_model
from ._person import DEAD, EXPOSED, INFECTED, RECOVERED, SUSCEPTIBLE, PersonElement
from ._person_view import PersonView
from ._traveller import TravellerElement
from ._traveller_view import TravellerView
from ._walker import DECEASED, HEALTHY, SICK, WalkerElement
from ._walker_view import WalkerView


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


def _is_living_walker(agent: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether an agent is a living walker, so it walks."""
    return isinstance(agent, WalkerView) and agent.health != DECEASED


def _get_walker_step(agent: AgentViewProtocol, destination: object, model: ModelViewProtocol) -> float:
    """Return the probability `travel` that a walker takes its step."""
    return float(agent.travel) if isinstance(agent, WalkerView) else 0.0


def _get_walker_infection(agent: AgentViewProtocol, other: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return the probability that a walker of the cell infects a healthy walker, `infection` when it is sick."""
    return float(other.infection) if isinstance(other, WalkerView) and other.health == SICK else 0.0


def _get_traveller_infection(agent: AgentViewProtocol, other: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return the probability that a traveller of the node infects a susceptible one, `tau` when it is infected."""
    return float(other.tau) if isinstance(other, TravellerView) and other.health == INFECTED else 0.0


def _is_due(agent: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return true, so a member moves as soon as it has held its value for the residence time."""
    return True


def _is_spreading(agent: PersonView, model: ModelView) -> bool:
    """Return whether an infectious person spreads in this step, drawn once per step with probability `beta`.

    The infection keeps the draw in its storage for the step, so the person infects every susceptible person of its cell
    or none of them.
    """
    record = model.storage.get(f'spreading.{agent.id}')
    if isinstance(record, tuple) and record[0] == model.time:
        return bool(record[1])
    spreads = bool(model.rng.random() < agent.beta)
    model.storage[f'spreading.{agent.id}'] = (model.time, spreads)
    return spreads


def _is_person_infected(agent: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return whether an infectious person of the cell of a susceptible person spreads to it in this step."""
    if not isinstance(agent, PersonView) or not isinstance(model, ModelView):
        return False
    return any(
        isinstance(other, PersonView) and other.health == INFECTED and _is_spreading(other, model)
        for other in agent.agents_here(agent.group)
    )


def _get_person_recovery(agent: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return the probability `gamma` that an infectious person recovers in a step."""
    return float(agent.gamma) if isinstance(agent, PersonView) else 0.0


def _get_person_death(agent: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return the probability `mu / (1 - gamma)` that an infectious person who has not recovered in a step dies.

    The death follows the recovery in the step, so this conditional probability makes the death `mu` of the step.
    """
    if not isinstance(agent, PersonView) or agent.gamma >= 1:
        return 0.0
    return float(agent.mu / (1 - agent.gamma))


def _get_person_waning(agent: AgentViewProtocol, model: ModelViewProtocol) -> float:
    """Return the probability `alpha` that a recovered person loses immunity in a step."""
    return float(agent.alpha) if isinstance(agent, PersonView) else 0.0


def _is_incubated(agent: AgentViewProtocol, model: ModelViewProtocol) -> bool:
    """Return true, so an exposed person becomes infectious as soon as its incubation is over."""
    return True


def _get_person_transitions(incubation: float) -> list[TransitionInteraction]:
    """Return the infection, waning immunity, recovery, death, and incubation of the people of the agent-based SIR."""
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
            condition=_is_person_infected,
        ),
        TransitionInteraction(
            'waning',
            target='people',
            state='health',
            schedule='member',
            form='agent',
            source=RECOVERED,
            destination=SUSCEPTIBLE,
            probability=_get_person_waning,
        ),
        TransitionInteraction(
            'recovery',
            target='people',
            state='health',
            schedule='member',
            form='agent',
            source=INFECTED,
            destination=RECOVERED,
            probability=_get_person_recovery,
        ),
        TransitionInteraction(
            'death',
            target='people',
            state='health',
            schedule='member',
            form='agent',
            source=INFECTED,
            destination=DEAD,
            probability=_get_person_death,
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
                condition=_is_incubated,
                after=incubation,
            ),
        )
    return transitions


def _initial_health(size: int, sick: int, rng: np.random.Generator, value: float) -> np.ndarray:
    """Return the health of a group with a number of randomly chosen sick members."""
    health = np.zeros(size)
    health[rng.choice(size, size=sick, replace=False)] = value
    return health


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
            'lockdown_start': lockdown_start,
            'lockdown_length': lockdown_length,
            'lockdown_d': lockdown_d,
        },
        location=[tuple(int(item) for item in cell) for cell in cells],
    )
    walk = MovementInteraction('walk', target='people', condition=_is_living_person, accept=_get_person_step)
    space = LatticeSpace((width, height), boundary='fixed', neighborhood='von_neumann')
    return System(elements=[group], interactions=[walk, *_get_person_transitions(incubation)], space=space)


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
        values={'health': health, 'infection': infection, 'travel': travel},
        location=locations,
    )
    interactions = [
        MovementInteraction(
            'walk',
            target='walkers',
            condition=_is_living_walker,
            accept=_get_walker_step,
            schedule='member',
        ),
        TransitionInteraction(
            'infection',
            target='walkers',
            state='health',
            source=HEALTHY,
            destination=SICK,
            probability=_get_walker_infection,
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
            after=math.ceil(lifetime),
            schedule='member',
            form='agent',
        ),
    ]
    return System(elements=[group], interactions=interactions, space=space)


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
        values={'health': _initial_health(size, 1, rng, INFECTED), 'tau': tau, 'm': m},
        location=[node for node in nodes for _ in range(residents)],
    )
    infection = TransitionInteraction(
        'infection',
        target='travellers',
        state='health',
        source=SUSCEPTIBLE,
        destination=INFECTED,
        probability=_get_traveller_infection,
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
        after=math.floor(days) + 1,
        schedule='member',
        form='agent',
    )
    travel = MovementInteraction('travel', target='travellers', accept=_get_traveller_step)
    return System(elements=[group], interactions=[infection, recovery, travel], space=NetworkSpace(graph))
