"""Evolutionary computation models: a genetic algorithm that evolves Robby the can-collecting robot."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.elements import Element, GroupElement
from ....core.interactions import EvolutionInteraction, Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import RoleSpec, StateSpec, VisualSpec
from ....types import Array, IntArray
from ..._catalog import register_model
from ._strategy_view import StrategyView

N_SITUATIONS = 243
N_ACTIONS = 7
EMPTY, CAN, WALL = 0, 1, 2
STAY, PICK_UP, RANDOM_MOVE = 4, 5, 6
MOVES = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
CAN_REWARD, EMPTY_PICK_UP_PENALTY, WALL_PENALTY = 10.0, -1.0, -5.0


def _new_rooms(rng: np.random.Generator, shape: tuple[int, ...], side: int, density: float) -> IntArray:
    """Return square rooms surrounded by walls, with a can in every inner cell with a probability."""
    rooms = np.full((*shape, side + 2, side + 2), WALL, dtype=np.int64)
    rooms[..., 1:-1, 1:-1] = (rng.random((*shape, side, side)) < density).astype(np.int64)
    return rooms


def _situations(rooms: IntArray, rows: IntArray, columns: IntArray) -> IntArray:
    """Return the index of what Robby sees north, south, east, west, and here, in base 3."""
    index = np.indices(rows.shape)
    seen = [rooms[(*index, rows + row, columns + column)] for row, column in [*MOVES.tolist(), [0, 0]]]
    return sum(value * 3**power for power, value in enumerate(seen))


def _evaluate(
    strategies: IntArray,
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


class StrategyElement(Element):
    """Strategy of Robby, the action for each of the 243 situations he can see, with its last measured score.

    The actions are 0 to 3 for moving north, south, east, and west, 4 to stay, 5 to pick up a can, and 6 to move in a
    random direction.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        actions:
            The initial action per situation, an array of 243 integers from 0 to 6.

        score:
            The initial score.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import StrategyElement
        >>> GroupElement('strategies', element=StrategyElement, size=3).get_params()['size']
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'score'}),
        *,
        actions: Annotated[object, StateSpec()] = 0.0,
        score: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.actions = actions
        self.score = score

    @classmethod
    def _get_view(cls) -> type[StrategyView]:
        """Return the view of the strategies, which types their states."""
        return StrategyView


class RoomCleaningInteraction(Interaction):
    """Score of every strategy of Robby, the mean of its scores in sessions of cleaning random rooms.

    Every strategy cleans `sessions` random rooms of side `side`, with a can in each cell with probability `density`,
    for `n_actions` actions, and its state `score` takes the mean score. A picked can scores 10, picking up nothing
    scores -1, and bumping into a wall scores -5.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of strategies.

        sessions:
            The number of rooms a strategy cleans to be scored.

        n_actions:
            The number of actions in a session.

        side:
            The side of a room.

        density:
            The probability that a cell holds a can.

        phase:
            The phase of the step the scoring runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.computer_science.evolutionary_computation import RoomCleaningInteraction
        >>> RoomCleaningInteraction('scoring', target='strategies').sessions
        10
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        sessions: int = 10,
        n_actions: int = 100,
        side: int = 10,
        density: float = 0.5,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.sessions = sessions
        self.n_actions = n_actions
        self.side = side
        self.density = density
        self.phase = phase

    def _score_agents(self, model: ModelView) -> None:
        """Set the score of every strategy of the target to its mean score in random rooms."""
        agents = model.agents('target', StrategyView)
        strategies = np.array([agent.actions for agent in agents], dtype=np.int64)
        for agent, score in zip(agents, self.score(strategies, model.rng), strict=True):
            agent.score = float(score)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that scores the strategies.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The scoring rule.
        """
        return [RuleLaw(self._score_agents, phase=self.phase)]

    def score(self, strategies: IntArray, rng: np.random.Generator) -> Array:
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

    Each step is one generation. Parents are selected with probabilities proportional to their rank, every pair of
    parents exchanges the tails of their strategies after a random cut, and every action of a child mutates to another
    action with probability `mutation`. Then every child is scored in random rooms. Random strategies score far below
    zero because they bump into walls, and evolution finds strategies that avoid walls and collect cans.

    Args:
        n_strategies:
            The size of the population.

        mutation:
            The probability that an action of a child is replaced by another one.

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
    evolution = EvolutionInteraction(
        'evolution',
        target='strategies',
        strategy='actions',
        fitness='score',
        selection='rank',
        crossover=1.0,
        mutation=mutation,
        alphabet=N_ACTIONS,
    )
    scoring = RoomCleaningInteraction('scoring', target='strategies', sessions=sessions, n_actions=n_actions)
    actions = rng.integers(0, N_ACTIONS, size=(n_strategies, N_SITUATIONS))
    scores = scoring.score(actions, rng)
    return System(
        elements=[
            GroupElement(
                'strategies',
                element=StrategyElement,
                size=n_strategies,
                values={'actions': actions, 'score': scores},
            ),
        ],
        interactions=[evolution, scoring],
    )
