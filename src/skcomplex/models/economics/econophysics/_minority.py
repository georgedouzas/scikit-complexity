"""Econophysics models: the minority game and its evolutionary version."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import IntArray
from ..._catalog import register_model
from ._game_state_spec import _GameStateSpec
from ._speculator_view import SpeculatorView

MAX_SEED = 2**31


def strategy_table(code: float, memory: int) -> IntArray:
    """Return the table of a minority game strategy, the bit it plays after each of the `2^m` histories.

    A table has `2^m` bits, too many to store as one number beyond small memories, so a strategy is stored as the
    integer code that seeds a generator of its table.

    Args:
        code:
            The integer code of the strategy.

        memory:
            The memory `m`, the number of past outcomes the strategy reads.

    Returns:
        table:
            The bit of the strategy for every history, from 0 to `2^m - 1`.

    Examples:
        >>> from skcomplex.models.economics.econophysics import strategy_table
        >>> strategy_table(7, 2).shape
        (4,)
    """
    return np.random.default_rng(int(code)).integers(0, 2, 2**memory)


class SpeculatorElement(Element):
    """Agent of the minority game, who holds two strategies and plays the one with the higher virtual score.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        first:
            The code of the first strategy.

        second:
            The code of the second strategy.

        first_score:
            The virtual score of the first strategy, the number of times it would have been in the minority.

        second_score:
            The virtual score of the second strategy.

        action:
            The last action, `+1` or `-1`.

        wealth:
            The net number of times the agent was in the minority.

    Examples:
        >>> from skcomplex.models.economics.econophysics import SpeculatorElement
        >>> GroupElement('players', element=SpeculatorElement, size=11, values={'first': 3, 'second': 5}).size
        11
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        first: Annotated[float, StateSpec(domain=NN)] = 0,
        second: Annotated[float, StateSpec(domain=NN)] = 1,
        first_score: Annotated[float, StateSpec()] = 0.0,
        second_score: Annotated[float, StateSpec()] = 0.0,
        action: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
        wealth: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.first = first
        self.second = second
        self.first_score = first_score
        self.second_score = second_score
        self.action = action
        self.wealth = wealth

    @classmethod
    def _get_view(cls) -> type[SpeculatorView]:
        """Return the view of the agents of the element."""
        return SpeculatorView


class MinorityGameInteraction(Interaction):
    """Rounds of the minority game of Challet and Zhang, where the agents on the minority side win.

    Every round each agent plays the action, `+1` or `-1`, that its best strategy assigns to the last `m` winning
    sides. The side with fewer agents wins, every agent on it gains one point and every other loses one, and every
    strategy gains one virtual point when it would have played the winning side. The volatility `sigma^2 / N` of the
    attendance `A = sum_i a_i` has a minimum below the value of random play near `alpha = 2^m / N ~ 0.34`. In the
    evolutionary version, every `period` rounds the poorest fraction of the agents replace their strategies by new
    random ones and restart with no wealth.

    Args:
        label:
            The label of the interaction.

        players:
            The path of the group of agents.

        fraction:
            The fraction of poorest agents that take new strategies at an evolution.

        memory:
            The memory `m`, the number of past winning sides the strategies read.

        period:
            The number of rounds between two evolutions, or 0 for no evolution.

    Examples:
        >>> from skcomplex.models.economics.econophysics import MinorityGameInteraction
        >>> MinorityGameInteraction('game', players='players', memory=4).memory
        4
    """

    reports: ClassVar[tuple[str, ...]] = ('attendance',)

    def __init__(
        self,
        label: str,
        *,
        players: Annotated[str, RoleSpec()],
        fraction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        memory: int = 3,
        period: int = 0,
    ) -> None:
        super().__init__(label)
        self.players = players
        self.fraction = fraction
        self.memory = memory
        self.period = period

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of a round.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.play)]

    def _table(self, code: float, model: ModelView) -> IntArray:
        """Return the table of a strategy, cached for the run in the state of the game."""
        state = self._state(model)
        if code not in state.tables:
            state = replace(state, tables={**state.tables, code: strategy_table(code, self.memory)})
            model.storage['game'] = state
        return state.tables[code]

    def _state(self, model: ModelView) -> _GameStateSpec:
        """Return the state of the game in the storage of the model, starting a random history at the first round."""
        state = model.storage.get('game')
        if not isinstance(state, _GameStateSpec):
            state = _GameStateSpec(int(model.rng.integers(2**self.memory)))
            model.storage['game'] = state
        return state

    def _choose(self, agent: SpeculatorView, history: int, model: ModelView) -> tuple[int, int]:
        """Set the action of an agent and return the actions of its two strategies."""
        actions = (
            2 * int(self._table(agent.first, model)[history]) - 1,
            2 * int(self._table(agent.second, model)[history]) - 1,
        )
        difference = agent.first_score - agent.second_score
        best = int(model.rng.integers(2)) if difference == 0 else int(difference < 0)
        agent.action = float(actions[best])
        return actions

    def play(self, model: ModelView) -> None:
        """Play one round, update the scores and the history, and evolve the agents when it is time.

        Args:
            model:
                The model.
        """
        agents = model.agents('players', SpeculatorView)
        n_histories = 2**self.memory
        drawn = int(model.rng.integers(n_histories))
        state = model.storage.get('game')
        if not isinstance(state, _GameStateSpec):
            model.storage['game'] = _GameStateSpec(drawn)
        history = self._state(model).history
        strategies = [self._choose(agent, history, model) for agent in agents]
        attendance = sum(agent.action for agent in agents)
        winner = -1 if attendance > 0 else 1
        for agent, (first, second) in zip(agents, strategies, strict=True):
            agent.first_score += first * winner
            agent.second_score += second * winner
            agent.wealth += agent.action * winner
        state = self._state(model)
        state = replace(state, history=(2 * history + (winner > 0)) % n_histories, round=state.round + 1)
        model.storage['game'] = state
        model.report('attendance', attendance)
        if self.period and state.round % self.period == 0:
            self._evolve(agents, model)

    def _evolve(self, agents: list[SpeculatorView], model: ModelView) -> None:
        """Give the poorest agents new random strategies and no wealth."""
        wealth = np.array([agent.wealth for agent in agents], dtype=float)
        count = round(model.parameter('fraction') * len(agents))
        for index in np.argsort(wealth, kind='stable')[:count]:
            agent = agents[int(index)]
            agent.first, agent.second = (float(code) for code in model.rng.integers(MAX_SEED, size=2))
            agent.first_score = agent.second_score = agent.wealth = 0.0


@register_model(
    field='economics/econophysics',
    tags=('minority game', 'agents', 'inductive reasoning', 'evolution'),
    references=('Challet and Zhang 1997', 'Yakovenko and Rosser 2009', 'Carmichael and Hadzikadic 2019 ch. 9'),
    defaults={'until': 500},
)
def minority_game(
    n_agents: int = 101,
    *,
    memory: int = 3,
    evolutionary: bool = False,
    period: int = 50,
    fraction: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the minority game of an odd number of agents with two random strategies each.

    Args:
        n_agents:
            The number of agents, odd so that there is always a minority.

        memory:
            The memory `m` of the strategies, so `alpha = 2^m / N`.

        evolutionary:
            Whether every `period` rounds the poorest agents take new random strategies.

        period:
            The number of rounds between two evolutions of the evolutionary version.

        fraction:
            The fraction of poorest agents that take new strategies at an evolution.

        random_state:
            The seed or generator of the strategies.

    Returns:
        system:
            The model, with the group `players` and the interaction `game`, which reports the attendance
            `game.attendance`.

    Raises:
        ValueError:
            The number of agents is even.

    Examples:
        >>> from skcomplex.models.economics.econophysics import minority_game
        >>> minority_game(11, memory=2).interactions[0].memory
        2
    """
    if n_agents % 2 == 0:
        error_msg = f'The minority game needs an odd number of agents, got {n_agents}.'
        raise ValueError(error_msg)
    codes = np.random.default_rng(random_state).integers(MAX_SEED, size=(2, n_agents)).astype(float)
    players = GroupElement(
        'players',
        element=SpeculatorElement,
        size=n_agents,
        values={'first': codes[0], 'second': codes[1]},
    )
    period = period if evolutionary else 0
    game = MinorityGameInteraction('game', players='players', memory=memory, period=period, fraction=fraction)
    return System(elements=[players], interactions=[game])
