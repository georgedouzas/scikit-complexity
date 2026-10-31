"""Tournament models: the evolution of strategies of the iterated prisoner's dilemma."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import EvolutionInteraction, GroupingInteraction
from ....core.system import System
from ....core.types import AgentViewProtocol, ModelViewProtocol
from ....specs import RoleSpec, StateSpec, VisualSpec
from ..._catalog import register_model
from ._strategist_view import StrategistView

GENES = 7
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


class StrategistElement(Element):
    """Agent of the prisoner's dilemma tournament, with a genome of responses and the fitness of its last games.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        genome:
            The seven moves, 1 to cooperate and 0 to defect, for the opponent's last two moves: none, C, D, CC, CD,
            DC, and DD.

        fitness:
            The mean score per round of the last tournament.

    Examples:
        >>> from skcomplex.models.social.cooperation import StrategistElement
        >>> group = GroupElement('agents', element=StrategistElement, size=10, values={'genome': [1.0] * 7})
        >>> group.label, group.element.__name__, group.size
        ('agents', 'StrategistElement', 10)
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        genome: Annotated[npt.ArrayLike, StateSpec(vector=True)] = 1.0,
        fitness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.genome = genome
        self.fitness = fitness

    @classmethod
    def _get_view(cls) -> type[StrategistView]:
        """Return the class of the views of the agents, whose attributes are the states and parameters."""
        return StrategistView


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


@register_model(
    field='social/cooperation',
    tags=('agents', 'games', 'cooperation', 'evolution'),
    references=('Axelrod 1984', 'Downey 2018'),
    defaults={'until': 50},
)
def prisoners_dilemma_tournament(
    n_agents: int = 100,
    *,
    rounds: int = 6,
    death: float = 0.1,
    mutation: float = 0.05,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the evolution of strategies of the iterated prisoner's dilemma from random genomes.

    Args:
        n_agents:
            The number of agents.

        rounds:
            The number of rounds of every game.

        death:
            The probability that an agent dies every step.

        mutation:
            The probability that an offspring has one gene flipped.

        random_state:
            The seed or generator of the initial genomes.

    Returns:
        system:
            The model, with the group `agents`, the tournament `tournament`, and the death-birth replacement
            `selection`.

    Examples:
        >>> from skcomplex.models.social.cooperation import prisoners_dilemma_tournament
        >>> prisoners_dilemma_tournament(10, random_state=0).elements[0].check().prepare().size_
        10
    """
    genomes = np.random.default_rng(random_state).integers(0, 2, (n_agents, GENES)).astype(float)
    visual = VisualSpec('histograms', {'color_by': 'fitness'})
    agents = GroupElement(
        'agents',
        element=StrategistElement,
        size=n_agents,
        values={'genome': genomes},
        visual=visual,
    )
    return System(
        elements=[agents],
        interactions=[
            TournamentInteraction('tournament', target='agents', rounds=rounds),
            EvolutionInteraction(
                'selection',
                target='agents',
                strategy='genome',
                selection='proportional',
                replacement='death-birth',
                mutation=mutation,
                death=death,
                change='one',
                parents='all',
            ),
        ],
    )
