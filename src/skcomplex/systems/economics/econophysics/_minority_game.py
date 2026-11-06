"""Rounds of the minority game of Challet and Zhang, where the agents on the minority side win."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import RoleSpec
from ....core.types import IntArray
from ._game_state_spec import _GameStateSpec
from ._minority_player_view import MinorityPlayerView


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
        >>> from skcomplex.systems.economics.econophysics import strategy_table
        >>> strategy_table(7, 2).shape
        (4,)
    """
    return np.random.default_rng(int(code)).integers(0, 2, 2**memory)


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
        >>> from skcomplex.systems.economics.econophysics import MinorityGameInteraction
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

    def _table(self, code: float, system: SystemView) -> IntArray:
        """Return the table of a strategy, cached for the run in the state of the game."""
        state = self._state(system)
        if code not in state.tables:
            state = replace(state, tables={**state.tables, code: strategy_table(code, self.memory)})
            system.storage['game'] = state
        return state.tables[code]

    def _state(self, system: SystemView) -> _GameStateSpec:
        """Return the state of the game in the storage of the system, starting a random history at the first round."""
        state = system.storage.get('game')
        if not isinstance(state, _GameStateSpec):
            state = _GameStateSpec(int(system.rng.integers(2**self.memory)))
            system.storage['game'] = state
        return state

    def play(self, system: SystemView) -> None:
        """Play one round, pay the agents, and update and report the history.

        Args:
            system:
                The system.
        """
        agents = system.agents('players', MinorityPlayerView)
        history = self._state(system).history
        for agent in agents:
            code = agent.second if agent.strategy else agent.first
            agent.action = float(2 * int(self._table(code, system)[history]) - 1)
        attendance = sum(agent.action for agent in agents)
        winner = -1 if attendance > 0 else 1
        for agent in agents:
            agent.wealth += agent.action * winner
        state = self._state(system)
        system.storage['game'] = replace(state, history=(2 * history + (winner > 0)) % 2**self.memory)
        system.report('attendance', attendance)
        system.report('history', float(history))

    def payoff(self, agent: MinorityPlayerView, system: SystemView, option: int) -> float:
        """Return the virtual payoff of a strategy of an agent in the last round, `+1` if it played the winning side.

        Args:
            agent:
                The agent.

            system:
                The system view of the rule that reads the payoff.

            option:
                The strategy, 0 for the first and 1 for the second.

        Returns:
            payoff:
                `+1` when the strategy would have played the winning side, or `-1`.
        """
        history = int(system.history(f'{self.label}.history')[-1])
        winner = -1 if system.history(f'{self.label}.attendance')[-1] > 0 else 1
        code = agent.second if option else agent.first
        return float((2 * int(strategy_table(code, self.memory)[history]) - 1) * winner)
