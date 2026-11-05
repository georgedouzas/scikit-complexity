"""Diffusion systems: beliefs and innovations that spread between neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.interactions import MapInteraction, MovementInteraction
from ....core.components.spaces import ContinuousSpace, LatticeSpace
from ....core.components.system import System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.types import Array
from ..._catalog import register_system
from ._adopter import AdopterElement
from ._adoption import AdoptionInteraction
from ._believer import BelieverElement
from ._believer_view import BelieverView

N_LEVELS = 4


PASSIONATE = 3


INNOVATORS = 0.025


def _shift(agent: AgentViewProtocol, system: SystemViewProtocol, size: float) -> Array:
    """Return the position one unit away from an agent in a random direction, wrapped on a square torus."""
    angle = system.rng.uniform(0, 2 * np.pi)
    position = np.asarray(agent.location, dtype=float) + np.array([np.cos(angle), np.sin(angle)])
    return np.mod(position, size)


def _learn(agent: BelieverView, system: SystemViewProtocol) -> Array:
    """Return the weights of an agent plus the influence of every neighbour on the level the neighbour cares at."""
    belief = np.array(agent.belief, dtype=np.float64)
    for neighbor in (item for item in agent.neighbors() if isinstance(item, BelieverView)):
        weights = np.asarray(neighbor.belief, dtype=np.float64)
        levels = np.flatnonzero(weights == weights.max())
        belief[levels[int(system.rng.integers(len(levels)))]] += agent.learning * neighbor.influence
    return belief


def _care(agent: BelieverView, system: SystemViewProtocol) -> float:
    """Return the level an agent cares at, the index of its largest weight."""
    return float(np.argmax(agent.belief))


@register_system(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'grid', 'influentials'),
    references=('Chesney 2021',),
    defaults={'until': 20},
)
def influentials(
    width: int = 20,
    height: int = 20,
    *,
    n_influentials: int = 10,
    influence: float = 3.0,
    news_report: bool = False,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the belief diffusion system with influentials who care passionately and weigh more with their neighbours.

    Every cell of a torus holds one agent whose weights start uniform on 0 to 4. Every agent in turn, in a random
    order, learns from its neighbours and then sets its level of caring.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        n_influentials:
            The number of influentials, chosen at random.

        influence:
            The influence of an influential, the others having influence 1.

        news_report:
            Whether a news report adds one to every agent's weight of caring passionately at the start.

        random_state:
            The seed or generator of the initial beliefs and of the influentials.

    Returns:
        system:
            The system, with the group `people`.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import influentials
        >>> influentials(5, 4, n_influentials=2, random_state=0).elements[0].prepare().size_
        20
    """
    rng = np.random.default_rng(random_state)
    size = width * height
    belief = rng.integers(0, 5, (size, N_LEVELS)).astype(float)
    belief[:, PASSIONATE] += float(news_report)
    chosen = rng.choice(size, n_influentials, replace=False)
    belief[chosen, PASSIONATE] = belief[chosen].max(axis=1) + 1
    weights = np.ones(size)
    weights[chosen] = influence
    people = GroupElement(
        'people',
        element=BelieverElement,
        size=size,
        values={'belief': belief, 'caring': belief.argmax(axis=1).astype(float), 'influence': weights},
        location=[(column, row) for column in range(width) for row in range(height)],
    )
    learning = MapInteraction(
        'learning',
        target='people',
        state='belief',
        function=_learn,
        schedule='member',
        form='agent',
    )
    caring = MapInteraction('caring', target='people', state='caring', function=_care, schedule='member', form='agent')
    return System(
        elements=[people],
        interactions=[learning, caring],
        space=LatticeSpace((width, height), boundary='periodic', capacity=1),
    )


@register_system(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'innovation', 'threshold'),
    references=('Granovetter 1978', 'Hamill and Gilbert 2016'),
    defaults={'until': 25},
)
def threshold_adoption(
    n_agents: int = 300,
    *,
    size: float = 175.0,
    reach: float = 15.0,
    mode: str = 'infection',
    thresholds: str = 'uniform',
    shifting: float = 0.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the threshold adoption system in social circles, started by the 2.5% of agents with the lowest thresholds.

    Args:
        n_agents:
            The number of agents, placed at random on a square torus.

        size:
            The side of the torus.

        reach:
            The distance within which other agents form an agent's social circle.

        mode:
            `'infection'` to adopt after one adopter in the circle, or `'influence'` to adopt above a threshold.

        thresholds:
            `'uniform'` for thresholds uniform on 1 to 100 percent, or `'normal'` for a normal distribution of mean 50
            and standard deviation 50 clipped to 1 to 100.

        shifting:
            The share of agents that move one unit in a random direction every step.

        random_state:
            The seed or generator of the positions and thresholds.

    Returns:
        system:
            The system, with the group `people`.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import threshold_adoption
        >>> threshold_adoption(40, size=60.0, random_state=0).elements[0].prepare().size_
        40
    """
    rng = np.random.default_rng(random_state)
    drawn = rng.uniform(1, 100, n_agents) if thresholds == 'uniform' else np.clip(rng.normal(50, 50, n_agents), 1, 100)
    innovators = np.argsort(drawn)[: max(1, round(INNOVATORS * n_agents))]
    adopted = np.zeros(n_agents)
    adopted[innovators] = 1.0
    people = GroupElement(
        'people',
        element=AdopterElement,
        size=n_agents,
        values={'adopted': adopted, 'threshold': drawn, 'reach': reach},
        location=rng.uniform(0, size, (n_agents, 2)),
    )
    return System(
        elements=[people],
        interactions=[
            MovementInteraction(
                'shift',
                target='people',
                step=lambda agent, system: _shift(agent, system, size),
                probability=shifting,
            ),
            AdoptionInteraction('adoption', target='people', mode=mode),
        ],
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
    )
