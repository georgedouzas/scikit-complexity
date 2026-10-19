"""Evolutionary computation models: a genetic algorithm that evolves Robby the can-collecting robot."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import AgentType, Interaction, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import ModelView

type Array = npt.NDArray[np.float64]
type Integers = npt.NDArray[np.int64]

N_SITUATIONS = 243
N_ACTIONS = 7
EMPTY, CAN, WALL = 0, 1, 2
STAY, PICK_UP, RANDOM_MOVE = 4, 5, 6
MOVES = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
CAN_REWARD, EMPTY_PICK_UP_PENALTY, WALL_PENALTY = 10.0, -1.0, -5.0


def _new_rooms(rng: np.random.Generator, shape: tuple[int, ...], side: int, density: float) -> Integers:
    """Return square rooms surrounded by walls, with a can in every inner cell with a probability."""
    rooms = np.full((*shape, side + 2, side + 2), WALL, dtype=np.int64)
    rooms[..., 1:-1, 1:-1] = (rng.random((*shape, side, side)) < density).astype(np.int64)
    return rooms


def _situations(rooms: Integers, rows: Integers, columns: Integers) -> Integers:
    """Return the index of what Robby sees north, south, east, west, and here, in base 3."""
    index = np.indices(rows.shape)
    seen = [rooms[(*index, rows + row, columns + column)] for row, column in [*MOVES.tolist(), [0, 0]]]
    return sum(value * 3**power for power, value in enumerate(seen))


def _evaluate(
    strategies: Integers,
    rng: np.random.Generator,
    *,
    sessions: int,
    actions: int,
    side: int,
    density: float,
) -> Array:
    """Return the mean score of every strategy over sessions of cleaning random rooms."""
    shape = (strategies.shape[0], sessions)
    rooms = _new_rooms(rng, shape, side, density)
    rows, columns = np.ones(shape, dtype=np.int64), np.ones(shape, dtype=np.int64)
    scores = np.zeros(shape)
    index = np.indices(shape)
    for _ in range(actions):
        action = strategies[index[0], _situations(rooms, rows, columns)]
        action = np.where(action == RANDOM_MOVE, rng.integers(0, 4, shape), action)
        here = rooms[(*index, rows, columns)]
        picking = action == PICK_UP
        scores += np.where(picking, np.where(here == CAN, CAN_REWARD, EMPTY_PICK_UP_PENALTY), 0.0)
        rooms[(*index, rows, columns)] = np.where(picking, EMPTY, here)
        moving = action < STAY
        step = MOVES[np.where(moving, action, 0)] * moving[..., None]
        target_rows, target_columns = rows + step[..., 0], columns + step[..., 1]
        blocked = rooms[(*index, target_rows, target_columns)] == WALL
        scores += np.where(blocked, WALL_PENALTY, 0.0)
        rows, columns = np.where(blocked, rows, target_rows), np.where(blocked, columns, target_columns)
    return scores.mean(axis=1)


def _offspring(strategies: Integers, scores: Array, rng: np.random.Generator, mutation: float) -> Integers:
    """Return children of parents chosen by rank, by single-point crossover and per-gene mutation."""
    size = strategies.shape[0]
    ranks = np.empty(size)
    ranks[np.argsort(scores)] = np.arange(1, size + 1)
    parents = rng.choice(size, size=(size, 2), p=ranks / ranks.sum())
    cuts = rng.integers(1, N_SITUATIONS, size=(size, 1))
    first = np.arange(N_SITUATIONS) < cuts
    children = np.where(first, strategies[parents[:, 0]], strategies[parents[:, 1]])
    mutated = rng.random(children.shape) < mutation
    return np.where(mutated, rng.integers(0, N_ACTIONS, children.shape), children)


class Strategy(AgentType):
    """Strategy of Robby, the action for each of the 243 situations he can see, with its last measured score.

    The actions are 0 to 3 for moving north, south, east, and west, 4 to stay, 5 to pick up a can, and 6 to move in a
    random direction.

    Args:
        label:
            The label of the agent type or group.

        location:
            Unused, since strategies have no space.

        actions:
            The initial action per situation, an array of 243 integers from 0 to 6.

        score:
            The initial score.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import Strategy
        >>> Strategy.group('strategies', 3)
        Group('strategies', element=Strategy, size=3)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'score'})
    actions: object = state(default=0.0)
    score: float = state(default=0.0)


class Evolution(Interaction):
    """Generation of a genetic algorithm: score every strategy in random rooms and breed the next generation.

    Every strategy cleans `sessions` random rooms of side `side`, with a can in each cell with probability `density`,
    for `n_actions` actions. A picked can scores 10, picking up nothing scores -1, and bumping into a wall scores -5.
    Parents are chosen with probabilities proportional to their rank, and every child takes a single-point crossover of
    two parents, with each action replaced by a random one with probability `mutation`.

    Args:
        label:
            The label of the interaction.

        mutation:
            The probability that an action of a child is replaced by a random one.

        population:
            The path of the group of strategies.

        sessions:
            The number of rooms a strategy cleans to be scored.

        n_actions:
            The number of actions in a session.

        side:
            The side of a room.

        density:
            The probability that a cell holds a can.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import Evolution
        >>> Evolution('evolution').sessions
        10
    """

    mutation: float = parameter(0.005, domain=Domain.NONNEGATIVE)
    population: str = 'strategies'
    sessions: int = 10
    n_actions: int = 100
    side: int = 10
    density: float = 0.5

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule of one generation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.breed)]

    def score(self, strategies: Integers, rng: np.random.Generator) -> Array:
        """Return the mean score of strategies in random rooms.

        Args:
            strategies:
                The actions of every strategy, one row per strategy.

            rng:
                The random number generator of the rooms and the random moves.

        Returns:
            scores:
                The mean score of every strategy.
        """
        return _evaluate(
            strategies,
            rng,
            sessions=self.sessions,
            actions=self.n_actions,
            side=self.side,
            density=self.density,
        )

    def breed(self, model: ModelView) -> None:
        """Replace the strategies by the next generation and score it.

        Args:
            model:
                The model.
        """
        agents = model.agents(self.population)
        strategies = np.array([agent.actions for agent in agents], dtype=np.int64)
        scores = np.array([agent.score for agent in agents], dtype=float)
        children = _offspring(strategies, scores, model.rng, model.parameters[f'{self.label}.mutation'])
        for agent, actions, score in zip(agents, children, self.score(children, model.rng), strict=True):
            agent.actions = actions
            agent.score = float(score)


@register_model(
    field='computer_science/evolutionary_computation',
    tags=('genetic algorithm', 'agents', 'evolution', 'robot'),
    references=('Mitchell 2009', 'Gooding 2019'),
    defaults={'until': 20},
)
def robby_ga(
    n_strategies: int = 50,
    *,
    mutation: float = 0.005,
    sessions: int = 10,
    n_actions: int = 100,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return Mitchell's genetic algorithm that evolves strategies for Robby, a robot that collects cans in a room.

    Each step is one generation. Random strategies score far below zero because they bump into walls, and evolution
    finds strategies that avoid walls and collect cans.

    Args:
        n_strategies:
            The size of the population.

        mutation:
            The probability that an action of a child is replaced by a random one.

        sessions:
            The number of rooms a strategy cleans to be scored.

        n_actions:
            The number of actions in a session.

        random_state:
            The seed or generator of the random first generation and of its scores.

    Returns:
        system:
            The model, with the group `strategies` of states `actions` and `score`.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import robby_ga
        >>> robby_ga(5).elements[0].values['actions'].shape
        (5, 243)
    """
    rng = np.random.default_rng(random_state)
    evolution = Evolution('evolution', mutation=mutation, sessions=sessions, n_actions=n_actions)
    actions = rng.integers(0, N_ACTIONS, size=(n_strategies, N_SITUATIONS))
    scores = evolution.score(actions, rng)
    return System(
        elements=[Strategy.group('strategies', n_strategies, actions=actions, score=scores)],
        interactions=[evolution],
    )
