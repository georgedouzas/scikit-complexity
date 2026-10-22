"""Opinion dynamics models: voting and majority rules."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import ClassVar

import networkx as nx
import numpy as np
import numpy.typing as npt

from ....core.base import Domain, VisualSpec, parameter, state
from ....core.model import (
    AgentType,
    Element,
    LatticeSpace,
    Law,
    LocalRule,
    Neighborhood,
    NetworkSpace,
    Rule,
    Symbols,
    System,
    Totalistic,
)
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

type Array = npt.NDArray[np.float64]

MAJORITY = 5


class Voter(AgentType):
    """Agent of the voter model, who adopts the opinion of a random neighbour on a network.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial node, or `None` for one agent per node in node order.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Voter
        >>> Voter.group('voters', 20, opinion=1.0)
        Group('voters', element=Voter, size=20)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)})
    opinion: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the imitation rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.imitate)]

    def imitate(self, agent: AgentView, model: ModelView) -> None:
        """Adopt the opinion of a random neighbour.

        Args:
            agent:
                The voter.

            model:
                The model.
        """
        neighbors = agent.neighbors()
        if neighbors:
            agent.opinion = neighbors[int(model.rng.integers(len(neighbors)))].opinion


class MajorityVoter(Element):
    """Cell that takes the majority opinion of its Moore neighbourhood, itself included.

    Args:
        label:
            The label of the cell or the lattice group.

        opinion:
            The initial opinion, 0 or 1.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import MajorityVoter
        >>> MajorityVoter('cell', opinion=1).opinion
        1
    """

    opinion: float = state(default=0, domain=Domain.BOOLEAN)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the majority rule over the nine cells.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            LocalRule(s.opinion, Neighborhood.moore(center=True), Totalistic(lambda total: total >= MAJORITY)),
        ]


class VariantVoter(AgentType):
    """Agent of the voter model variants: an imitator, a threshold voter, or a contrarian.

    With a threshold of 0 the agent copies a random neighbour, as in the voter model. With a positive threshold it
    adopts the opinion opposite to its own when the share of neighbours that hold it exceeds the threshold, so a
    threshold of 0.5 gives the majority rule. A contrarian adopts the opposite of what it would otherwise adopt.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial node, or `None` for one agent per node in node order.

        opinion:
            The initial opinion, 0 or 1.

        contrarian:
            Whether the agent is a contrarian, 1, or not, 0.

        threshold:
            The share of disagreeing neighbours above which the agent changes its opinion, or 0 to copy a random
            neighbour.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import VariantVoter
        >>> VariantVoter.group('voters', 10, threshold=0.5)
        Group('voters', element=VariantVoter, size=10)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)})
    opinion: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    contrarian: float = state(default=0.0, domain=Domain.BOOLEAN)
    threshold: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the voting rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.vote)]

    def _choice(self, agent: AgentView, neighbors: list[AgentView], model: ModelView) -> float:
        """Return the opinion an agent that conforms would take."""
        if agent.threshold <= 0:
            return float(neighbors[int(model.rng.integers(len(neighbors)))].opinion)
        disagreeing = sum(neighbor.opinion != agent.opinion for neighbor in neighbors) / len(neighbors)
        return float(1 - agent.opinion) if disagreeing > agent.threshold else float(agent.opinion)

    def vote(self, agent: AgentView, model: ModelView) -> None:
        """Take the opinion of the agent's variant, inverted for a contrarian.

        Args:
            agent:
                The voter.

            model:
                The model.
        """
        neighbors = agent.neighbors()
        if not neighbors:
            return
        choice = self._choice(agent, neighbors, model)
        agent.opinion = 1 - choice if agent.contrarian else choice


def _random_cells(shape: Sequence[int], density: float, random_state: int | np.random.Generator | None) -> Array:
    """Return cells that are 1 with a probability and 0 otherwise."""
    return (np.random.default_rng(random_state).random(tuple(shape)) < density).astype(float)


@register_model(
    field='social/opinion_dynamics',
    tags=('lattice', 'opinion', 'voting'),
    references=('Charbonneau 2017',),
    defaults={'until': 50, 'dt': 1},
)
def majority_rule(
    shape: Sequence[int] = (100, 100),
    *,
    density: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return voters that take the majority opinion of their neighbourhood, which coarsens into domains.

    Args:
        shape:
            The number of rows and columns.

        density:
            The probability that a voter starts with opinion 1.

        random_state:
            The seed or generator of the initial opinions.

    Returns:
        system:
            The model, with the opinions `voters.opinion`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import majority_rule
        >>> majority_rule((10, 10)).elements[0].size
        100
    """
    values = _random_cells(shape, density, random_state)
    return System(components=[MajorityVoter.lattice('voters', LatticeSpace(tuple(shape)), opinion=values)])


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'network'),
    references=('Holley and Liggett 1975',),
    defaults={'until': 100},
)
def voter_model(
    graph: object = None,
    *,
    share: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return a voter model with one voter per node of a network.

    Args:
        graph:
            The NetworkX graph, or `None` for a ring of 50 nodes each linked to its four nearest neighbours.

        share:
            The share of voters with opinion 1.

        random_state:
            The seed or generator of the opinions.

    Returns:
        system:
            The model, with the group `voters`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.social.opinion_dynamics import voter_model
        >>> voter_model(nx.cycle_graph(8), random_state=0).elements[0].size
        8
    """
    rng = np.random.default_rng(random_state)
    graph = nx.watts_strogatz_graph(50, 4, 0.0) if graph is None else graph
    size = nx.number_of_nodes(graph)
    opinions = (rng.random(size) < share).astype(float)
    return System(components=[Voter.group('voters', size, opinion=opinions)], space=NetworkSpace(graph))


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'network', 'voting', 'contrarians'),
    references=('Thurner, Hanel, and Klimek 2018', 'Galam 2004'),
    defaults={'until': 100},
)
def voter_variants(
    graph: object = None,
    *,
    threshold: float = 0.5,
    contrarians: float = 0.1,
    share: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return voter model variants on a network, with threshold voters and a share of contrarians.

    Args:
        graph:
            The NetworkX graph, or `None` for a ring of 50 nodes each linked to its four nearest neighbours.

        threshold:
            The share of disagreeing neighbours above which a voter changes its opinion, 0.5 for the majority rule, or
            0 to copy a random neighbour.

        contrarians:
            The share of voters that are contrarians.

        share:
            The share of voters with opinion 1.

        random_state:
            The seed or generator of the opinions and of the contrarians.

    Returns:
        system:
            The model, with the group `voters`.

    Examples:
        >>> import networkx as nx
        >>> from skcomplex.models.social.opinion_dynamics import voter_variants
        >>> voter_variants(nx.cycle_graph(8), random_state=0).elements[0].size
        8
    """
    rng = np.random.default_rng(random_state)
    graph = nx.watts_strogatz_graph(50, 4, 0.0) if graph is None else graph
    size = nx.number_of_nodes(graph)
    voters = VariantVoter.group(
        'voters',
        size,
        opinion=(rng.random(size) < share).astype(float),
        contrarian=(rng.random(size) < contrarians).astype(float),
        threshold=threshold,
    )
    return System(components=[voters], space=NetworkSpace(graph))
