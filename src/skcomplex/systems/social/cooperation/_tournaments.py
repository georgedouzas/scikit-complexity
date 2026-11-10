"""Tournament systems: the evolution of strategies of the iterated dilemma.

The iterated dilemma is the prisoner's dilemma.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import GroupingInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import RoleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ...general.agents import EvolutionInteraction

PAIR = 2
PAYOFFS = {(1, 1): (3.0, 3.0), (1, 0): (0.0, 5.0), (0, 1): (5.0, 0.0), (0, 0): (1.0, 1.0)}
GENES = 7


class StrategistView(AgentView):
    """View of a strategist of the iterated prisoner's dilemma.

    Its states are attributes.
    """

    genome: Array
    fitness: float


class StrategistElement(Element):
    """Agent of the prisoner's dilemma tournament.

    It holds a genome of responses and the fitness of its last games.

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
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.social.cooperation import StrategistElement
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
        """Return the class of the views of the agents.

        Their attributes are the states and parameters.
        """
        return StrategistView


def _gene(history: list[int]) -> int:
    """Return the gene that answers the last two moves of the opponent.

    The moves are none, C, D, CC, CD, DC, or DD.
    """
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
        >>> from skcomplex.systems.social.cooperation import play
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
        >>> from skcomplex.systems.social.cooperation import TournamentInteraction
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

    def _get_groups(self, system: SystemViewProtocol) -> list[list[AgentViewProtocol]]:
        """Return every agent of a random ring followed by its two neighbours in the ring."""
        agents = [agent for agent in system.get_agents('target') if agent.alive]
        if len(agents) < PAIR:
            return []
        ring = [agents[int(index)] for index in system.rng.permutation(len(agents))]
        return [[agent, ring[position - 1], ring[(position + 1) % len(ring)]] for position, agent in enumerate(ring)]

    def act_on(self, members: list[AgentViewProtocol], system: SystemViewProtocol) -> None:
        """Set the fitness of the first member to its mean score per round.

        The score is against the two others.

        Args:
            members:
                The agent, followed by its two neighbours in the ring.

            system:
                The system view of the rule.
        """
        agent, *opponents = [member for member in members if isinstance(member, StrategistView)]
        total = sum(play(agent.genome, opponent.genome, self.rounds)[0] for opponent in opponents)
        agent.fitness = total / (PAIR * self.rounds)


class PrisonersDilemmaTournamentSystem(BaseSystem):
    """The evolution of strategies of the iterated prisoner's dilemma.

    The strategies start from random genomes. The system has the group `agents`, the tournament `tournament`, and the
    death-birth replacement `selection`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.cooperation import PrisonersDilemmaTournamentSystem
        >>> PrisonersDilemmaTournamentSystem(10, random_state=0).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'social/cooperation'
    topics = ('agents', 'games', 'cooperation', 'evolution')
    references = ('Axelrod 1984', 'Downey 2018')
    until = 50

    def __init__(
        self,
        n_agents: int = 100,
        *,
        rounds: int = 6,
        death: float = 0.1,
        mutation: float = 0.05,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_agents = n_agents
        self.rounds = rounds
        self.death = death
        self.mutation = mutation
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        genomes = np.random.default_rng(self.random_state).integers(0, 2, (self.n_agents, GENES)).astype(float)
        visual = VisualSpec('histograms', {'color_by': 'fitness'})
        agents = GroupElement(
            'agents',
            element=StrategistElement,
            size=self.n_agents,
            values={'genome': genomes},
            visual=visual,
        )
        return CompositeSystem(
            self.label,
            elements=[agents],
            interactions=[
                TournamentInteraction('tournament', target='agents', rounds=self.rounds),
                EvolutionInteraction(
                    'selection',
                    target='agents',
                    strategy='genome',
                    selection='proportional',
                    replacement='death-birth',
                    mutation=self.mutation,
                    death=self.death,
                    change='one',
                    parents='all',
                ),
            ],
        )
