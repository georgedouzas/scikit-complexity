"""Tournament models: the evolution of strategies of the iterated prisoner's dilemma."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.model import AgentType, Interaction, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

type Array = npt.NDArray[np.float64]

GENES = 7
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


def _offspring(parent: AgentView, rng: np.random.Generator) -> Array:
    """Return the genome of a parent, with one gene flipped with the mutation probability."""
    genome = np.array(parent.genome, dtype=float)
    if rng.random() < parent.mutation:
        index = int(rng.integers(GENES))
        genome[index] = 1.0 - genome[index]
    return genome


class Strategist(AgentType):
    """Agent of the prisoner's dilemma tournament, with a genome of responses and the fitness of its last games.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        genome:
            The seven moves, 1 to cooperate and 0 to defect, for the opponent's last two moves: none, C, D, CC, CD,
            DC, and DD.

        fitness:
            The mean score per round of the last tournament.

        death:
            The probability that an agent dies every step.

        mutation:
            The probability that an offspring has one gene flipped.

    Examples:
        >>> from skcomplex.models.social.cooperation import Strategist
        >>> Strategist.group('agents', 10, genome=[1.0] * 7)
        Group('agents', element=Strategist, size=10)
    """

    genome: npt.ArrayLike = state(default=1.0, vector=True)
    fitness: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    death: float = parameter(default=0.1, domain=Domain.NONNEGATIVE)
    mutation: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)


class Tournament(Interaction):
    """Tournament and selection of strategies of the iterated prisoner's dilemma.

    Every step the agents stand in a random ring and play their two neighbours, and their fitness is their mean score
    per round. Then every agent dies with probability `death` and is replaced by the offspring of an agent chosen with
    probability proportional to its fitness.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of strategists.

        rounds:
            The number of rounds of every game.

    Examples:
        >>> from skcomplex.models.social.cooperation import Tournament
        >>> Tournament('tournament', target='agents').rounds
        6
    """

    target: str = role()
    rounds: int = 6

    def laws(self, s: Symbols) -> list[Law]:
        """Return the tournament rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.compete)]

    def _melee(self, agents: list[AgentView], rng: np.random.Generator) -> Array:
        """Return the mean score per round of every agent after playing its two neighbours in a random ring."""
        order = rng.permutation(len(agents))
        totals = np.zeros(len(agents))
        for position, first in enumerate(order):
            second = order[(position + 1) % len(order)]
            scores = play(agents[first].genome, agents[second].genome, self.rounds)
            totals[first] += scores[0]
            totals[second] += scores[1]
        return totals / (2 * self.rounds)

    def compete(self, model: ModelView) -> None:
        """Play the tournament, then replace the dead with offspring chosen by fitness.

        Args:
            model:
                The model.
        """
        agents = model.agents(self.target)
        if len(agents) < 2:  # noqa: PLR2004
            return
        fitness = self._melee(agents, model.rng)
        dead = np.flatnonzero(model.rng.random(len(agents)) < agents[0].death)
        parents = model.rng.choice(len(agents), size=len(dead), p=fitness / fitness.sum())
        genomes = [_offspring(agents[parent], model.rng) for parent in parents]
        for agent, value in zip(agents, fitness, strict=True):
            agent.fitness = float(value)
        for index, genome in zip(dead, genomes, strict=True):
            agents[index].genome = genome


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
            The model, with the group `agents`.

    Examples:
        >>> from skcomplex.models.social.cooperation import prisoners_dilemma_tournament
        >>> prisoners_dilemma_tournament(10, random_state=0).elements[0].size
        10
    """
    genomes = np.random.default_rng(random_state).integers(0, 2, (n_agents, GENES)).astype(float)
    visual = VisualSpec('traits', {'color_by': 'fitness'})
    agents = Strategist.group('agents', n_agents, genome=genomes, death=death, mutation=mutation, visual=visual)
    return System(components=[agents], interactions=[Tournament('tournament', target='agents', rounds=rounds)])
