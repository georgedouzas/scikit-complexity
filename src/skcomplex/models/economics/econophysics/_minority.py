"""Econophysics models: the minority game and its evolutionary version."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import NN, ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.elements import Element, GroupElement
from ....core.interactions import ChoiceInteraction, EvolutionInteraction, Interaction, ReinforcementInteraction
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import DiscreteTime
from ....core.system import System
from ....engine.solvers import ModelView
from ....specs import PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ....types import IntArray
from ..._catalog import register_model
from ._game_state_spec import _GameStateSpec
from ._minority_player_view import MinorityPlayerView

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


class MinorityPlayerElement(Element):
    """Agent of the minority game, who holds two strategies and plays the one it chose.

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

        strategy:
            The strategy played, 0 for the first and 1 for the second.

        action:
            The last action, `+1` or `-1`.

        wealth:
            The net number of times the agent was in the minority.

    Examples:
        >>> from skcomplex.models.economics.econophysics import MinorityPlayerElement
        >>> GroupElement('players', element=MinorityPlayerElement, size=11, values={'first': 3, 'second': 5}).size
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
        strategy: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        action: Annotated[float, StateSpec(domain=ZZ)] = 0.0,
        wealth: Annotated[float, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.first = first
        self.second = second
        self.first_score = first_score
        self.second_score = second_score
        self.strategy = strategy
        self.action = action
        self.wealth = wealth

    @classmethod
    def _get_view(cls) -> type[MinorityPlayerView]:
        """Return the view of the agents of the element."""
        return MinorityPlayerView


class MinorityGameInteraction(Interaction):
    """Rounds of the minority game of Challet and Zhang, where the agents on the minority side win.

    Every round each agent plays the action, `+1` or `-1`, that its chosen strategy assigns to the last `m` winning
    sides, the strategy of higher virtual score that a `ChoiceInteraction` of rule `'best'` chooses before. The side
    with fewer agents wins, and every agent on it gains one point and every other loses one. The game reports the
    attendance `A = sum_i a_i` and the code of the history the strategies read, so a cumulative
    `ReinforcementInteraction` gives every strategy one virtual point when it would have played the winning side, by
    the payoff `payoff`. The volatility `sigma^2 / N` of the attendance has a minimum below the value of random play
    near `alpha = 2^m / N ~ 0.34`.

    Args:
        label:
            The label of the interaction.

        players:
            The path of the group of agents.

        memory:
            The memory `m`, the number of past winning sides the strategies read.

        phase:
            The phase the rounds run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.economics.econophysics import MinorityGameInteraction
        >>> MinorityGameInteraction('game', players='players', memory=4).memory
        4
    """

    reports: ClassVar[tuple[str, ...]] = ('attendance', 'history')

    def __init__(
        self,
        label: str,
        *,
        players: Annotated[str, RoleSpec()],
        memory: int = 3,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.players = players
        self.memory = memory
        self.phase = phase

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of a round.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.play, phase=self.phase)]

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

    def play(self, model: ModelView) -> None:
        """Play one round, pay the agents, and update and report the history.

        Args:
            model:
                The model.
        """
        agents = model.agents('players', MinorityPlayerView)
        history = self._state(model).history
        for agent in agents:
            code = agent.second if agent.strategy else agent.first
            agent.action = float(2 * int(self._table(code, model)[history]) - 1)
        attendance = sum(agent.action for agent in agents)
        winner = -1 if attendance > 0 else 1
        for agent in agents:
            agent.wealth += agent.action * winner
        state = self._state(model)
        model.storage['game'] = replace(state, history=(2 * history + (winner > 0)) % 2**self.memory)
        model.report('attendance', attendance)
        model.report('history', float(history))

    def payoff(self, agent: MinorityPlayerView, model: ModelView, option: int) -> float:
        """Return the virtual payoff of a strategy of an agent in the last round, `+1` if it played the winning side.

        Args:
            agent:
                The agent.

            model:
                The model view of the rule that reads the payoff.

            option:
                The strategy, 0 for the first and 1 for the second.

        Returns:
            payoff:
                `+1` when the strategy would have played the winning side, or `-1`.
        """
        history = int(model.history(f'{self.label}.history')[-1])
        winner = -1 if model.history(f'{self.label}.attendance')[-1] > 0 else 1
        code = agent.second if option else agent.first
        return float((2 * int(strategy_table(code, self.memory)[history]) - 1) * winner)


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
            The model, with the group `players`, the choice `choice` of the strategy of higher virtual score, the
            interaction `game`, which reports the attendance `game.attendance`, the cumulative reinforcement
            `scoring` of the virtual scores, and in the evolutionary version the truncation `evolution`, which gives
            the poorest agents new random strategies and no wealth every `period` rounds.

    Raises:
        ValueError:
            The number of agents is even.

    Examples:
        >>> from skcomplex.models.economics.econophysics import minority_game
        >>> minority_game(11, memory=2).interactions[1].memory
        2
    """
    if n_agents % 2 == 0:
        error_msg = f'The minority game needs an odd number of agents, got {n_agents}.'
        raise ValueError(error_msg)
    codes = np.random.default_rng(random_state).integers(MAX_SEED, size=(2, n_agents)).astype(float)
    players = GroupElement(
        'players',
        element=MinorityPlayerElement,
        size=n_agents,
        values={'first': codes[0], 'second': codes[1]},
    )
    choice = ChoiceInteraction(
        'choice',
        target='players',
        state='strategy',
        options=(0.0, 1.0),
        utility=lambda agent, model, option: agent.second_score if option else agent.first_score,
        rule='best',
        form='agent',
        phase='choose',
    )
    game = MinorityGameInteraction('game', players='players', memory=memory, phase='play')
    scoring = ReinforcementInteraction(
        'scoring',
        target='players',
        scores=('first_score', 'second_score'),
        payoff=game.payoff,
        rule='cumulative',
        form='agent',
        phase='score',
    )
    evolution = EvolutionInteraction(
        'evolution',
        target='players',
        strategy=('first', 'second'),
        fitness='wealth',
        replacement='truncation',
        fraction=fraction,
        mutation=1.0,
        change='any',
        alphabet=MAX_SEED,
        period=period,
        reset={'first_score': 0.0, 'second_score': 0.0, 'wealth': 0.0},
        phase='score',
    )
    return System(
        elements=[players],
        interactions=[choice, game, scoring, *([evolution] if evolutionary else [])],
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('choose'), PhaseSpec('play'), PhaseSpec('score')]),
        ),
    )
