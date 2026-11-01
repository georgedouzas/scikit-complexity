"""Tournament of strategies of the iterated prisoner's dilemma."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import numpy.typing as npt

from ....core.interactions import GroupingInteraction
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import RoleSpec
from ._strategist_view import StrategistView

PAIR = 2


PAYOFFS = {(1, 1): (3.0, 3.0), (1, 0): (0.0, 5.0), (0, 1): (5.0, 0.0), (0, 0): (1.0, 1.0)}


def _gene(history: list[int]) -> int:
    """Return the gene that answers the last two moves of the opponent: none, C, D, CC, CD, DC, or DD."""
    recent = history[-2:]
    if not recent:
        return 0
    if len(recent) == 1:
        return 1 if recent[0] else 2
    return 3 + 2 * (1 - recent[0]) + (1 - recent[1])


def play(first: npt.ArrayLike, second: npt.ArrayLike, rounds: int = 6) -> tuple[float, float]:
    """Return the scores of two genomes in an iterated prisoner's dilemma.

    A genome holds seven moves, 1 to cooperate and 0 to defect, for the opponent's last two moves: none, C, D, CC,
    CD, DC, and DD. A round pays 3 to both cooperators, 1 to both defectors, and 5 to a defector against a cooperator,
    who gets 0.

    Args:
        first:
            The genome of the first player.

        second:
            The genome of the second player.

        rounds:
            The number of rounds.

    Returns:
        scores:
            The total scores of the two players.

    Examples:
        >>> from skcomplex.models.social.cooperation import play
        >>> play([1] * 7, [0] * 7)
        (0.0, 30.0)
    """
    genomes = (np.asarray(first), np.asarray(second))
    moves: tuple[list[int], list[int]] = ([], [])
    scores = [0.0, 0.0]
    for _ in range(rounds):
        move = (int(genomes[0][_gene(moves[1])]), int(genomes[1][_gene(moves[0])]))
        payoff = PAYOFFS[move]
        scores = [scores[0] + payoff[0], scores[1] + payoff[1]]
        moves[0].append(move[0])
        moves[1].append(move[1])
    return scores[0], scores[1]


class TournamentInteraction(GroupingInteraction):
    """Tournament of strategies of the iterated prisoner's dilemma.

    Every step the agents stand in a random ring, and every agent forms a group with its two neighbours in the ring,
    whom it plays. Its fitness is its mean score per round.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of strategists.

        rounds:
            The number of rounds of every game.

        phase:
            The phase of the step the tournament runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.models.social.cooperation import TournamentInteraction
        >>> TournamentInteraction('tournament', target='agents').rounds
        6
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        rounds: int = 6,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, groups='neighborhood', phase=phase)
        self.rounds = rounds

    def _get_groups(self, model: ModelViewProtocol) -> list[list[AgentViewProtocol]]:
        """Return every agent of a random ring followed by its two neighbours in the ring."""
        agents = [agent for agent in model.agents('target') if agent.alive]
        if len(agents) < PAIR:
            return []
        ring = [agents[int(index)] for index in model.rng.permutation(len(agents))]
        return [[agent, ring[position - 1], ring[(position + 1) % len(ring)]] for position, agent in enumerate(ring)]

    def act_on(self, members: list[AgentViewProtocol], model: ModelViewProtocol) -> None:
        """Set the fitness of the first member to its mean score per round against the two others.

        Args:
            members:
                The agent, followed by its two neighbours in the ring.

            model:
                The model view of the rule.
        """
        agent, *opponents = [member for member in members if isinstance(member, StrategistView)]
        total = sum(play(agent.genome, opponent.genome, self.rounds)[0] for opponent in opponents)
        agent.fitness = total / (PAIR * self.rounds)
