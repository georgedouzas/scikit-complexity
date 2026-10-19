"""Model segregation, wealth exchange, and opinion dynamics with agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import networkx as nx
import numpy as np

from ...core.base import Domain, VisualSpec, parameter, state
from ...core.model import AgentType, GridSpace, Law, NetworkSpace, Rule, Symbols, System
from ...core.runtime import register_model
from ...core.solvers import AgentView, ModelView


class Resident(AgentType):
    """Resident of the Schelling model, who moves to an empty cell while too few neighbours share its kind.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, or `None` for a random empty cell.

        kind:
            The kind of the resident, 0 or 1.

        threshold:
            The smallest share of neighbours of the same kind that keeps a resident in place.

    Examples:
        >>> from skcomplex.models.social import Resident
        >>> Resident.group('residents', 10, kind=1.0)
        Group('residents', element=Resident, size=10)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'kind', 'color_range': (0.0, 1.0)})
    kind: float = state(default=0.0, domain=Domain.NONNEGATIVE_INTEGER)
    threshold: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the relocation rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.relocate)]

    def relocate(self, agent: AgentView, model: ModelView) -> None:
        """Move to a random empty cell while the share of similar neighbours is below the threshold.

        Args:
            agent:
                The resident.

            model:
                The model.
        """
        neighbors = agent.neighbors()
        if not neighbors:
            return
        similar = sum(neighbor.kind == agent.kind for neighbor in neighbors) / len(neighbors)
        if similar < agent.threshold:
            agent.move_to(model.random_empty_cell())


class Trader(AgentType):
    """Agent of the Boltzmann wealth model, who gives one unit of wealth to a random agent when it has any.

    Args:
        label:
            The label of the agent type or group.

        location:
            The initial location, unused without a space.

        wealth:
            The initial wealth.

    Examples:
        >>> from skcomplex.models.social import Trader
        >>> Trader.group('traders', 100)
        Group('traders', element=Trader, size=100)
    """

    wealth: float = state(default=1.0, domain=Domain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the exchange rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.give)]

    def give(self, agent: AgentView, model: ModelView) -> None:
        """Give one unit of wealth to a random agent of the same group.

        Args:
            agent:
                The trader.

            model:
                The model.
        """
        if agent.wealth <= 0:
            return
        others = model.agents(agent.group)
        receiver = others[int(model.rng.integers(len(others)))]
        receiver.wealth += 1
        agent.wealth -= 1


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
        >>> from skcomplex.models.social import Voter
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


@register_model(
    field='social',
    tags=('segregation', 'agents', 'grid'),
    references=('Schelling 1971',),
    defaults={'until': 50},
)
def schelling(
    width: int = 20,
    height: int = 20,
    *,
    density: float = 0.8,
    minority: float = 0.3,
    threshold: float = 0.5,
    random_state: int | np.random.Generator | None = None,
) -> System:
    """Return a Schelling segregation model on a grid with at most one resident per cell.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        density:
            The share of cells with a resident.

        minority:
            The share of residents of kind 1.

        threshold:
            The smallest share of similar neighbours that keeps a resident in place.

        random_state:
            The seed or generator of the kinds.

    Returns:
        system:
            The model, with the group `residents`.

    Examples:
        >>> from skcomplex.models.social import schelling
        >>> schelling(10, 10, density=0.5).elements[0].size
        50
    """
    rng = np.random.default_rng(random_state)
    size = int(width * height * density)
    kinds = (rng.random(size) < minority).astype(float)
    residents = Resident.group('residents', size, kind=kinds, threshold=threshold)
    return System(elements=[residents], space=GridSpace(width, height, capacity=1))


@register_model(
    field='social',
    tags=('wealth', 'agents', 'exchange'),
    references=('Dragulescu and Yakovenko 2000',),
    defaults={'until': 100},
)
def boltzmann_wealth(n_agents: int = 100) -> System:
    """Return a Boltzmann wealth exchange model where every agent starts with one unit.

    Args:
        n_agents:
            The number of agents.

    Returns:
        system:
            The model, with the group `traders`.

    Examples:
        >>> from skcomplex.models.social import boltzmann_wealth
        >>> boltzmann_wealth(10).elements[0].size
        10
    """
    return System(elements=[Trader.group('traders', n_agents)])


@register_model(
    field='social',
    tags=('opinion', 'agents', 'network'),
    references=('Holley and Liggett 1975',),
    defaults={'until': 100},
)
def voter_model(
    graph: object = None,
    *,
    share: float = 0.5,
    random_state: int | np.random.Generator | None = None,
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
        >>> from skcomplex.models.social import voter_model
        >>> voter_model(nx.cycle_graph(8), random_state=0).elements[0].size
        8
    """
    rng = np.random.default_rng(random_state)
    graph = nx.watts_strogatz_graph(50, 4, 0.0) if graph is None else graph
    size = nx.number_of_nodes(graph)
    opinions = (rng.random(size) < share).astype(float)
    return System(elements=[Voter.group('voters', size, opinion=opinions)], space=NetworkSpace(graph))
