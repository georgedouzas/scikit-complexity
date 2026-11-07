"""Network superposition model: a union of Erdős-Rényi subgraphs whose linking probabilities are drawn."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.spaces import NetworkSpace
from ....core.components.system import System
from ....core.errors import ValidationError
from ....core.specs import StateSpec, VisualSpec
from ....core.types import Array
from ..._catalog import register_system

DEGREE_VISUAL = VisualSpec('sites', {'states': ['degree']})


class SuperposedNodeElement(Element):
    """Node of a superposed network, with its degree and its clustering.

    Args:
        label:
            The label of the node group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        degree:
            The number of edges of the node.

        clustering:
            The share of the pairs of neighbours of the node that are joined.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import SuperposedNodeElement
        >>> SuperposedNodeElement('nodes').clustering
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = DEGREE_VISUAL,
        *,
        degree: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        clustering: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.degree = degree
        self.clustering = clustering


def _linking_probabilities(size: int, exponent: float, bounds: tuple[float, float], rng: np.random.Generator) -> Array:
    """Return linking probabilities drawn from `Q(p) ~ p^-exponent` between two bounds by inverse transform."""
    low, high = bounds
    uniform = rng.uniform(size=size)
    if exponent == 1:
        return low * (high / low) ** uniform
    power = 1 - exponent
    return (low**power + uniform * (high**power - low**power)) ** (1 / power)


@register_system(
    field='mathematics/networks',
    tags=('network', 'superposition', 'random graph', 'superstatistics', 'degree distribution'),
    references=('Abe and Thurner 2005', 'Thurner et al. 2018'),
    defaults={'until': 1, 'dt': 1},
)
def network_superposition(
    n_nodes: int = 100,
    *,
    n_subgraphs: int = 20,
    exponent: float = 2.0,
    low: float = 0.005,
    high: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the network superposition model, a union of Erdős-Rényi graphs whose linking probabilities are drawn.

    Every one of `n_subgraphs` subgraphs of `n_nodes` nodes draws one linking probability `p` from
    `Q(p) ~ p^-exponent` on `[low, high]` and links each pair of its nodes with probability `p`, and the network is
    their union (Thurner et al. 2018, section 4.5.2). Its degrees mix the Poisson laws of the subgraphs,
    `P(k) = ∫ dp P(k | N p) Q(p)`, Eq. 4.34, so a power law of the probabilities gives a power law of the degrees up
    to the largest mean degree `N high`. The book draws `p` for every attempted link, which gives one Erdős-Rényi graph
    of the mean probability, so the probability is drawn once per subgraph, the reading of Eq. 4.34, like the cells of
    superstatistics.

    Args:
        n_nodes:
            The number of nodes of every subgraph.

        n_subgraphs:
            The number of subgraphs.

        exponent:
            The exponent of the distribution `Q(p)` of the linking probabilities.

        low:
            The smallest linking probability.

        high:
            The largest linking probability.

        random_state:
            The seed or generator of the linking probabilities and the links.

    Returns:
        system:
            The system, with the group `nodes` and their states `degree` and `clustering`.

    Raises:
        ValidationError:
            The bounds of the linking probabilities are not `0 < low <= high <= 1`.

    Examples:
        >>> from skcomplex.systems.mathematics.networks import network_superposition
        >>> system = network_superposition(10, n_subgraphs=3)
        >>> system.elements[0].values['degree'].size
        30
    """
    if not 0 < low <= high <= 1:
        error_msg = f'The linking probabilities lie in [{low}, {high}]. Give bounds with 0 < low <= high <= 1.'
        raise ValidationError(error_msg)
    rng = np.random.default_rng(random_state)
    graph = nx.Graph()
    for index, chance in enumerate(_linking_probabilities(n_subgraphs, exponent, (low, high), rng)):
        subgraph = nx.fast_gnp_random_graph(n_nodes, float(chance), seed=rng)
        graph.update(nx.relabel_nodes(subgraph, {node: index * n_nodes + node for node in subgraph}))
    clustering = nx.clustering(graph)
    nodes = GroupElement(
        'nodes',
        element=SuperposedNodeElement,
        size=graph.number_of_nodes(),
        values={
            'degree': np.asarray([graph.degree(node) for node in graph], dtype=float),
            'clustering': np.asarray([clustering[node] for node in graph], dtype=float),
        },
        location=list(graph),
    )
    return System(elements=[nodes], space=NetworkSpace(graph))
