"""Score of every strategy of Robby, the mean of its scores in sessions of cleaning random rooms."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....engine.solvers import ModelView
from ....specs import RoleSpec
from ....types import Array, IntArray
from ._strategy_view import StrategyView

EMPTY, CAN, WALL = 0, 1, 2


STAY, PICK_UP, RANDOM_MOVE = 4, 5, 6


MOVES = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])


CAN_REWARD, EMPTY_PICK_UP_PENALTY, WALL_PENALTY = 10.0, -1.0, -5.0


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
        return RoomCleaningInteraction._evaluate(
            strategies,
            rng,
            sessions=self.sessions,
            actions=self.n_actions,
            side=self.side,
            density=self.density,
        )

    @staticmethod
    def _new_rooms(rng: np.random.Generator, shape: tuple[int, ...], side: int, density: float) -> IntArray:
        """Return square rooms surrounded by walls, with a can in every inner cell with a probability."""
        rooms = np.full((*shape, side + 2, side + 2), WALL, dtype=np.int64)
        rooms[..., 1:-1, 1:-1] = (rng.random((*shape, side, side)) < density).astype(np.int64)
        return rooms

    @staticmethod
    def _situations(rooms: IntArray, rows: IntArray, columns: IntArray) -> IntArray:
        """Return the index of what Robby sees north, south, east, west, and here, in base 3."""
        index = np.indices(rows.shape)
        seen = [rooms[(*index, rows + row, columns + column)] for row, column in [*MOVES.tolist(), [0, 0]]]
        return sum(value * 3**power for power, value in enumerate(seen))

    @staticmethod
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
        rooms = RoomCleaningInteraction._new_rooms(rng, shape, side, density)
        rows, columns = np.ones(shape, dtype=np.int64), np.ones(shape, dtype=np.int64)
        scores = np.zeros(shape)
        index = np.indices(shape)
        for _ in range(actions):
            action = strategies[index[0], RoomCleaningInteraction._situations(rooms, rows, columns)]
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
