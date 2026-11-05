"""Bounded confidence models: compromise between agents whose opinions are close, in pairs or all at once."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


import networkx as nx
import numpy as np

from ....core.elements import GroupElement, LatticeElement
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.system import System
from ....specs import VisualSpec
from ..._catalog import register_model
from ._confidence_coupling import ConfidenceCouplingInteraction
from ._confident import ConfidentElement
from ._discussant import DiscussantElement
from ._encounters import EncountersInteraction

PAIR = 2


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
        >>> deffuant(20, random_state=0).elements[0].prepare().size_
        20
    """
    opinions = np.random.default_rng(random_state).random(n_agents)
    visual = VisualSpec('histograms', {'color_by': 'opinion', 'color_range': (0.0, 1.0), 'histogram_first': True})
    people = GroupElement(
        'people',
        element=DiscussantElement,
        size=n_agents,
        values={'opinion': opinions, 'confidence': confidence, 'convergence': convergence},
        visual=visual,
    )
    return System(elements=[people], interactions=[EncountersInteraction('meetings', target='people')])


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
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.social.opinion_dynamics import hegselmann_krause
        >>> trajectory = Simulation(
        ...     hegselmann_krause(50, confidence=0.3), random_state=0,
        ... ).prepare().run(until=20).trajectory_
        >>> len(np.unique(np.round(trajectory.values('people.opinion')[-1], 6))) <= 3
        True
    """
    space = NetworkSpace(graph if graph is not None else nx.complete_graph(n_agents)).prepare()
    opinions = np.random.default_rng(random_state).random(space.size_)
    people = LatticeElement(
        'people',
        element=ConfidentElement,
        space=space,
        values={'opinion': opinions, 'confidence': confidence},
    )
    coupling = ConfidenceCouplingInteraction('compromise', target='people')
    return System(elements=[people], interactions=[coupling], time=DiscreteTime())
