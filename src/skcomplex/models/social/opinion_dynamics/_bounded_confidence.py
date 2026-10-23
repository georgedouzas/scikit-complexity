"""Bounded confidence models: compromise between agents whose opinions are close, in pairs or all at once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import networkx as nx
import numpy as np

from ....core.backends import ModelView
from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction
from ....core.laws import Law, Rule, Symbols, Update, neighbors, where
from ....core.runtime import register_model
from ....core.space import Adjacency, DiscreteTime, NetworkSpace
from ....core.system import System


class Discussant(Element):
    """Agent of the Deffuant model, with a continuous opinion and a confidence bound.

    Args:
        label:
            The label of the agent or group.

        opinion:
            The initial opinion, between 0 and 1.

        confidence:
            The largest opinion difference at which two agents still compromise.

        convergence:
            The fraction of the opinion difference by which each agent of a compromising pair moves.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Discussant
        >>> Discussant.group('people', 10, opinion=0.5)
        Group('people', element=Discussant, size=10)
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'opinion', 'color_range': (0.0, 1.0)})
    opinion: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    confidence: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)
    convergence: float = parameter(default=0.5, domain=Domain.NONNEGATIVE)


class Encounters(Interaction):
    """Random binary encounters of the Deffuant model, as many per step as there are agents.

    In every encounter two random agents whose opinions differ by less than the confidence bound move towards each
    other by the convergence fraction of their difference.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of discussants.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Encounters
        >>> Encounters('meetings', target='people').target
        'people'
    """

    target: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the encounter rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.meet)]

    def meet(self, model: ModelView) -> None:
        """Run one random encounter per agent.

        Args:
            model:
                The model.
        """
        agents = model.agents(self.target)
        if len(agents) < 2:  # noqa: PLR2004
            return
        confidence, convergence = agents[0].confidence, agents[0].convergence
        opinions = np.array([agent.opinion for agent in agents], dtype=float)
        pairs = model.rng.integers(len(agents), size=(len(agents), 2))
        for first, second in pairs:
            difference = opinions[second] - opinions[first]
            if first != second and abs(difference) < confidence:
                opinions[first] += convergence * difference
                opinions[second] -= convergence * difference
        for agent, opinion in zip(agents, opinions, strict=True):
            agent.opinion = float(opinion)


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'agents', 'bounded confidence', 'clusters'),
    references=('Deffuant et al. 2000', 'Castellano, Fortunato, and Loreto 2009', 'Gros 2015'),
    defaults={'until': 50},
)
def deffuant(
    n_agents: int = 200,
    *,
    confidence: float = 0.2,
    convergence: float = 0.5,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Deffuant bounded confidence model, whose opinions settle into about `1 / (2 * confidence)` clusters.

    Args:
        n_agents:
            The number of agents.

        confidence:
            The largest opinion difference at which two agents compromise.

        convergence:
            The fraction of the difference by which each agent of a compromising pair moves.

        random_state:
            The seed or generator of the uniform initial opinions.

    Returns:
        system:
            The model, with the group `people`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import deffuant
        >>> deffuant(20, random_state=0).elements[0].size
        20
    """
    opinions = np.random.default_rng(random_state).random(n_agents)
    visual = VisualSpec('traits', {'color_by': 'opinion', 'color_range': (0.0, 1.0), 'histogram_first': True})
    people = Discussant.group(
        'people',
        n_agents,
        opinion=opinions,
        confidence=confidence,
        convergence=convergence,
        visual=visual,
    )
    return System(components=[people], interactions=[Encounters('meetings', target='people')])


class Confident(Element):
    """Agent of the Hegselmann-Krause model, who moves to the mean opinion of the neighbours within its confidence.

    Every step all agents update at once to the mean opinion of the neighbours, themselves included, whose opinions
    differ from theirs by less than `confidence`.

    Args:
        label:
            The label of the agent or group.

        opinion:
            The opinion, between 0 and 1.

        confidence:
            The largest opinion difference of a neighbour an agent still listens to.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Confident
        >>> Confident('people', confidence=0.1).confidence
        0.1
    """

    visual: ClassVar[VisualSpec] = VisualSpec('traits', {'color_range': (0.0, 1.0), 'histogram_first': True})
    opinion: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    confidence: float = parameter(default=0.2, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the move to the mean opinion of the confident neighbours.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        close = where(abs(s.other.opinion - s.opinion) < s.confidence, 1, 0)
        mean = neighbors(s.other.opinion, over=Adjacency(center=True), reduce='normalized-mean', weight=close)
        return [Update(s.opinion, mean)]


@register_model(
    field='social/opinion_dynamics',
    tags=('opinion', 'bounded confidence', 'clusters', 'network', 'synchronous'),
    references=('Hegselmann and Krause 2002', 'Castellano, Fortunato, and Loreto 2009'),
    defaults={'until': 30},
)
def hegselmann_krause(
    n_agents: int = 200,
    *,
    confidence: float = 0.15,
    graph: nx.Graph | None = None,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Hegselmann-Krause model, whose opinions freeze within a few steps into clusters about `2 ε` apart.

    Every step every agent moves at once to the mean opinion of its neighbours, itself included, whose opinions lie
    within the confidence `ε` of its own. On the complete graph the opinions settle into about `1 / (2 ε)` clusters.

    Args:
        n_agents:
            The number of agents, the nodes of the complete graph when no graph is given.

        confidence:
            The confidence `ε`.

        graph:
            The network of the agents, or `None` for the complete graph.

        random_state:
            The seed or generator of the uniform initial opinions.

    Returns:
        system:
            The model, with the group `people` on the nodes of the network.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.runtime import Simulation
        >>> from skcomplex.models.social.opinion_dynamics import hegselmann_krause
        >>> trajectory = Simulation(hegselmann_krause(50, confidence=0.3), random_state=0).run(until=20).trajectory_
        >>> len(np.unique(np.round(trajectory['people.opinion'][-1], 6))) <= 3
        True
    """
    space = NetworkSpace(graph if graph is not None else nx.complete_graph(n_agents))
    opinions = np.random.default_rng(random_state).random(space.size)
    people = Confident.lattice('people', space, opinion=opinions, confidence=confidence)
    return System(components=[people], time=DiscreteTime())
