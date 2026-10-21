"""Econophysics models: the minority game and its evolutionary version."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, parameter, role, state
from ....core.model import AgentType, Interaction, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

type Bits = npt.NDArray[np.int64]

MAX_SEED = 2**31


def strategy_table(code: float, memory: int) -> Bits:
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


class Speculator(AgentType):
    """Agent of the minority game, who holds two strategies and plays the one with the higher virtual score.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

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
        >>> from skcomplex.models.economics.econophysics import Speculator
        >>> Speculator.group('players', 11, first=3, second=5).size
        11
    """

    first: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    second: float = state(default=1, domain=Domain.NONNEGATIVE_INTEGER)
    first_score: float = state(default=0.0, domain=Domain.REAL)
    second_score: float = state(default=0.0, domain=Domain.REAL)
    action: float = state(default=0.0, domain=Domain.INTEGER)
    wealth: float = state(default=0.0, domain=Domain.REAL)


class MinorityGame(Interaction):
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

        memory:
            The memory `m`, the number of past winning sides the strategies read.

        period:
            The number of rounds between two evolutions, or 0 for no evolution.

        fraction:
            The fraction of poorest agents that take new strategies at an evolution.

    Examples:
        >>> from skcomplex.models.economics.econophysics import MinorityGame
        >>> MinorityGame('game', players='players', memory=4).memory
        4
    """

    reports: ClassVar[tuple[str, ...]] = ('attendance',)
    players: str = role()
    memory: int = 3
    period: int = 0
    fraction: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule of a round.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.play)]

    def _table(self, code: float, model: ModelView) -> Bits:
        """Return the table of a strategy, cached for the run."""
        tables = model.storage.setdefault('tables', {})
        if code not in tables:
            tables[code] = strategy_table(code, self.memory)
        return tables[code]

    def _choose(self, agent: AgentView, history: int, model: ModelView) -> tuple[int, int]:
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
        agents = model.agents('players')
        storage = model.storage
        n_histories = 2**self.memory
        history = storage.setdefault('history', int(model.rng.integers(n_histories)))
        strategies = [self._choose(agent, history, model) for agent in agents]
        attendance = sum(agent.action for agent in agents)
        winner = -1 if attendance > 0 else 1
        for agent, (first, second) in zip(agents, strategies, strict=True):
            agent.first_score += first * winner
            agent.second_score += second * winner
            agent.wealth += agent.action * winner
        storage['history'] = (2 * history + (winner > 0)) % n_histories
        storage['round'] = storage.get('round', 0) + 1
        model.report('attendance', attendance)
        if self.period and storage['round'] % self.period == 0:
            self._evolve(agents, model)

    def _evolve(self, agents: list[AgentView], model: ModelView) -> None:
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
    players = Speculator.group('players', n_agents, first=codes[0], second=codes[1])
    period = period if evolutionary else 0
    game = MinorityGame('game', players='players', memory=memory, period=period, fraction=fraction)
    return System(elements=[players], interactions=[game])
