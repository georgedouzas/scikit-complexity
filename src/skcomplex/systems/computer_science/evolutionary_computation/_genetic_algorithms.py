"""Evolutionary computation systems: Mitchell's and Gooding's evolution of Robby.

They are Mitchell's genetic algorithm and Gooding's evolution by circumstance of Robby.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated, Protocol, cast

import numpy as np
from sage.all__sagemath_symbolics import NN

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.errors import ParameterError
from ....core.systems import BaseSystem, System
from ....core.types import Array, IntArray, Role
from ...general.agents import EvolutionInteraction

N_SITUATIONS = 243
N_ACTIONS = 7
REGIMES = ('generational', 'circumstance')
EMPTY, CAN, WALL = 0, 1, 2
STAY, PICK_UP, RANDOM_MOVE = 4, 5, 6
MOVES = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
CAN_REWARD, EMPTY_PICK_UP_PENALTY, WALL_PENALTY = 10.0, -1.0, -5.0


class StrategyView(AgentViewProtocol, Protocol):
    """View of a strategy of a genetic algorithm, with its states as attributes."""

    actions: IntArray
    score: float


class StrategyElement(Element):
    """Strategy of Robby, with its last measured score.

    The strategy gives the action for each of the 243 situations he can see. The actions are 0 to 3 for moving north,
    south, east, and west, 4 to stay, 5 to pick up a can, and 6 to move in a random direction.

    Args:
        label:
            The label of the agent or group.

        actions:
            The initial action per situation, an array of 243 integers from 0 to 6.

        score:
            The initial score.

        size:
            The number of strategies, or `None` for one.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import StrategyElement
        >>> StrategyElement('strategies', size=3).get_params()['size']
        3
    """

    visual = VisualSpec('agents', {'color_by': 'score'})

    def __init__(
        self,
        label: str,
        *,
        actions: object = 0.0,
        score: object = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.actions = actions
        self.score = score


class RoomCleaningInteraction(Interaction):
    """Score of every strategy of Robby.

    It is the mean of its scores in sessions of cleaning random rooms. Every strategy cleans `sessions` random rooms of
    side `side`, with a can in each cell with probability `density`, for `n_actions` actions, and its state `score`
    takes the mean score. A picked can scores 10, picking up nothing scores -1, and bumping into a wall scores -5.

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
        >>> from skcomplex.systems.computer_science.evolutionary_computation import RoomCleaningInteraction
        >>> RoomCleaningInteraction('scoring', target='strategies').sessions
        10
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
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

    def _score_agents(self, system: SystemView) -> None:
        """Set the score of every strategy of the target to its mean score in random rooms."""
        agents = cast(list[StrategyView], system.get_agents('target'))
        strategies = np.array([agent.actions for agent in agents], dtype=np.int64)
        for agent, score in zip(agents, self.score(strategies, system.rng), strict=True):
            agent.score = float(score)

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rule that scores the strategies, once per step for the whole population.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseComponent] = [UpdateLaw(context, self._score_agents, phase=self.phase)]

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
    def _build_rooms(rng: np.random.Generator, shape: tuple[int, ...], side: int, density: float) -> IntArray:
        """Return square rooms surrounded by walls.

        Every inner cell holds a can with a probability.
        """
        rooms = np.full((*shape, side + 2, side + 2), WALL, dtype=np.int64)
        rooms[..., 1:-1, 1:-1] = (rng.random((*shape, side, side)) < density).astype(np.int64)
        return rooms

    @staticmethod
    def _get_situations(rooms: IntArray, rows: IntArray, columns: IntArray) -> IntArray:
        """Return the index of what Robby sees.

        He sees north, south, east, west, and here, and the index is in base 3.
        """
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
        rooms = RoomCleaningInteraction._build_rooms(rng, shape, side, density)
        rows, columns = np.ones(shape, dtype=np.int64), np.ones(shape, dtype=np.int64)
        scores = np.zeros(shape)
        index = np.indices(shape)
        for _ in range(actions):
            action = strategies[index[0], RoomCleaningInteraction._get_situations(rooms, rows, columns)]
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


class AgedStrategyView(StrategyView):
    """View of a strategy of Robby with an age, with its states as attributes."""

    age: float


class AgedStrategyElement(StrategyElement):
    """Strategy of Robby with an age.

    It dies of old age or starves in the evolution by circumstance.

    Args:
        label:
            The label of the agent or group.

        actions:
            The initial action per situation, an array of 243 integers from 0 to 6.

        score:
            The initial score.

        age:
            The initial age in generations.

        size:
            The number of strategies, or `None` for one.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import AgedStrategyElement
        >>> AgedStrategyElement('strategy', age=3).age
        3
    """

    visual = VisualSpec('agents', {'color_by': 'score'})

    def __init__(
        self,
        label: str,
        *,
        actions: object = 0.0,
        score: object = 0.0,
        age: Annotated[object, NN] = 0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, actions=actions, score=score, size=size)
        self.age = age


class CircumstanceInteraction(Interaction):
    """Evolution by circumstance of the strategies of Robby.

    Strategies die of old age or starve, and random survivors breed in their places.

    Every generation the strategies age by one, the ones that reach `lifespan` die of old age, and the fraction
    `starvation` of the population with the lowest scores starves. Random pairs of survivors, whatever their score,
    exchange the tails of their strategies after a random cut, mutate every action to another one with probability
    `mutation`, and place their child, of age 0, in the place of a dead strategy, until the population is whole again.
    With a `switch` period, the starvation is on for `switch` generations and off for the next `switch`, in turn.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of aged strategies.

        lifespan:
            The age at which a strategy dies.

        starvation:
            The fraction of the population, of the lowest scores, that starves every generation.

        mutation:
            The probability that an action of a child is replaced by another one.

        switch:
            The number of generations after which the starvation switches on or off, or `None` to keep it on.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import CircumstanceInteraction
        >>> CircumstanceInteraction('circumstance', target='strategies').lifespan
        7
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        lifespan: int = 7,
        starvation: float = 0.3,
        mutation: float = 0.005,
        switch: int | None = None,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.lifespan = lifespan
        self.starvation = starvation
        self.mutation = mutation
        self.switch = switch

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rule of the deaths and births of a generation, once per step for the whole population.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        self.laws_: list[BaseComponent] = [UpdateLaw(context, self._advance_generation)]

    def _advance_generation(self, system: SystemView) -> None:
        """Age the strategies and replace the old and starving ones.

        Their places are filled with children of survivors.
        """
        agents = cast(list[AgedStrategyView], system.get_agents('target'))
        rng = system.rng
        ages = np.array([agent.age for agent in agents]) + 1
        scores = np.array([agent.score for agent in agents])
        dead = ages >= self.lifespan
        if self._is_starving(system.time):
            dead[np.argsort(scores, kind='stable')[: round(self.starvation * len(agents))]] = True
        survivors = np.flatnonzero(~dead)
        if not survivors.size:
            survivors = np.arange(len(agents))
        strategies = np.array([agent.actions for agent in agents], dtype=np.int64)
        for agent, age in zip(agents, ages, strict=True):
            agent.age = float(age)
        for index in np.flatnonzero(dead):
            first, second = rng.choice(survivors, size=2)
            agents[index].actions = self._get_child(strategies[first], strategies[second], rng)
            agents[index].age = 0.0

    def _is_starving(self, time: float) -> bool:
        """Return whether the starvation is on in the generation of a time."""
        return self.switch is None or int(time // self.switch) % 2 == 0

    def _get_child(self, first: IntArray, second: IntArray, rng: np.random.Generator) -> IntArray:
        """Return the child of two strategies.

        It takes the head of one and the tail of the other, with mutated actions.
        """
        cut = int(rng.integers(1, first.size))
        child = np.concatenate([first[:cut], second[cut:]])
        mutated = rng.random(child.size) < self.mutation
        shifts = rng.integers(1, N_ACTIONS, child.size)
        return np.where(mutated, (child + shifts) % N_ACTIONS, child)


class RobbyGASystem(BaseSystem):
    """Mitchell's genetic algorithm that evolves strategies for Robby.

    Robby is a robot that collects cans in a room. Each step is one generation. Parents are selected with probabilities
    proportional to their rank, every pair of parents exchanges the tails of their strategies after a random cut, and
    every action of a child mutates to another action with probability `mutation`. Then every child is scored in random
    rooms. Random strategies score far below zero because they bump into walls, and evolution finds strategies that
    avoid walls and collect cans.

    Gooding's evolution by circumstance, the regime `'circumstance'`, replaces the selection with deaths: every strategy
    starts at a random age, dies at the age `lifespan`, and the fraction `starvation` with the lowest scores starves
    every generation. Random pairs of survivors, whatever their scores, breed children in the places of the dead. The
    starvation is the only selection, so with it the scores rise as under the genetic algorithm, and without it they
    drift. With a `switch` period the starvation switches on and off every `switch` generations, as in Gooding's runs
    that switch it every 3000 iterations. The system has the group `strategies` of states `actions` and `score`, and
    `age` in the evolution by circumstance.

    Args:
        n_strategies:
            The size of the population.

        mutation:
            The probability that an action of a child is replaced by another one.

        sessions:
            The number of rooms a strategy cleans to be scored.

        n_actions:
            The number of actions in a session.

        regime:
            The evolution, `'generational'` for Mitchell's genetic algorithm or `'circumstance'` for Gooding's
            evolution by circumstance.

        lifespan:
            The age at which a strategy dies, in the evolution by circumstance.

        starvation:
            The fraction of the lowest scores that starves every generation, in the evolution by circumstance.

        switch:
            The number of generations after which the starvation switches on or off, or `None` to keep it on, in the
            evolution by circumstance.

        random_state:
            The seed or generator of the random first generation and of its scores.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The regime is unknown.

    Examples:
        >>> from skcomplex.systems.computer_science.evolutionary_computation import RobbyGASystem
        >>> RobbyGASystem(5).prepare().system_.elements[0].actions.shape
        (5, 243)
    """

    field = 'computer_science/evolutionary_computation'
    topics = ('genetic algorithm', 'agents', 'evolution', 'robot')
    references = ('Mitchell 2009', 'Gooding 2019')
    until = 20

    def __init__(
        self,
        n_strategies: int = 50,
        *,
        mutation: float = 0.005,
        sessions: int = 10,
        n_actions: int = 100,
        regime: str = 'generational',
        lifespan: int = 7,
        starvation: float = 0.3,
        switch: int | None = None,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_strategies = n_strategies
        self.mutation = mutation
        self.sessions = sessions
        self.n_actions = n_actions
        self.regime = regime
        self.lifespan = lifespan
        self.starvation = starvation
        self.switch = switch
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components of the system built from its model parameters."""
        super()._prepare(context)
        if self.regime not in REGIMES:
            error_msg = f'Unknown regime {self.regime!r}. Use one of {", ".join(REGIMES)}.'
            raise ParameterError(error_msg)
        rng = np.random.default_rng(self.random_state)
        evolution = EvolutionInteraction(
            'evolution',
            target='strategies',
            strategy='actions',
            fitness='score',
            selection='rank',
            crossover=1.0,
            mutation=self.mutation,
            alphabet=N_ACTIONS,
        )
        scoring = RoomCleaningInteraction(
            'scoring',
            target='strategies',
            sessions=self.sessions,
            n_actions=self.n_actions,
        )
        actions = rng.integers(0, N_ACTIONS, size=(self.n_strategies, N_SITUATIONS))
        scores = scoring.score(actions, rng)
        if self.regime == 'circumstance':
            circumstance = CircumstanceInteraction(
                'circumstance',
                target='strategies',
                lifespan=self.lifespan,
                starvation=self.starvation,
                mutation=self.mutation,
                switch=self.switch,
            )
            ages = rng.integers(0, self.lifespan, size=self.n_strategies).astype(float)
            strategies = AgedStrategyElement(
                'strategies',
                actions=actions,
                score=scores,
                age=ages,
                size=self.n_strategies,
            )
            self.system_ = System(self.label, elements=[strategies], interactions=[circumstance, scoring])
            return
        self.system_ = System(
            self.label,
            elements=[StrategyElement('strategies', actions=actions, score=scores, size=self.n_strategies)],
            interactions=[evolution, scoring],
        )
