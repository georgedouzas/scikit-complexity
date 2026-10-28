"""Growing networks: preferential attachment, triad formation, and vertex copying."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

import networkx as nx
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, UpdateLaw
from ....core.laws.utils import neighbors, total, where
from ....core.spaces import DiscreteTime, NetworkSpace
from ....core.specs import ParameterSpec, StateSpec, TopologyRequestSpec, VisualSpec
from ....core.system import System
from ..._catalog import register_model

DEGREE_VISUAL = VisualSpec('sites', {'states': ['degree']})
SMALLEST_TOTAL = 1e-12
SMALLEST_DEGREE = 1e-9


def _degree(node: SimpleNamespace) -> object:
    """Return the degree of a candidate node, its weight under preferential attachment."""
    return neighbors(1)


class LinkElement(Element):
    """Edge of a growing network, which holds no state of its own.

    Args:
        label:
            The label of the edge group.

    Examples:
        >>> from skcomplex.models.mathematics.networks import LinkElement
        >>> LinkElement('links').label
        'links'
    """


class VertexElement(Element):
    """Node of a growing network, with its age in steps and its degree.

    New nodes start with age 0, and the degree is read from the edges at the start of every step.

    Args:
        label:
            The label of the node group.

        age:
            The number of steps since the node joined.

        degree:
            The number of edges of the node.

    Examples:
        >>> from skcomplex.models.mathematics.networks import VertexElement
        >>> VertexElement('nodes').age
        0.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = DEGREE_VISUAL,
        *,
        age: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        degree: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.age = age
        self.degree = degree

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the ageing and the degree of the node.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [UpdateLaw(s.age, s.age + 1), UpdateLaw(s.degree, neighbors(1))]


class TriadVertexElement(VertexElement):
    """Node of a Holme-Kim network, which closes triangles with the neighbours of its first contact when it joins.

    One step after a node joins with one edge by preferential attachment, it adds `extra` edges. Each new end is a
    neighbour of its first contact with probability `triad`, the triad formation step, and a node drawn by
    preferential attachment otherwise. The state `near` marks the nodes next to the newest node.

    Args:
        label:
            The label of the node group.

        near:
            The number of neighbours of the node that joined in the previous step.

        triad:
            The probability that a new edge closes a triangle.

        extra:
            The number of edges a node adds after its first one.

    Examples:
        >>> from skcomplex.models.mathematics.networks import TriadVertexElement
        >>> TriadVertexElement('nodes', triad=0.5).triad
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: float = 0.0,
        degree: float = 0.0,
        near: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        triad: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        extra: Annotated[int, ParameterSpec(domain=NN)] = 1,
    ) -> None:
        super().__init__(label, visual=visual, age=age, degree=degree)
        self.near = near
        self.triad = triad
        self.extra = extra

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the ageing, the degree, the marks next to the newest node, and the edges of triad formation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        triad = s.triad

        def weight(node: SimpleNamespace) -> object:
            """Return the mixture of triad formation and preferential attachment for a candidate node."""
            closing = neighbors(node.other.near)
            attaching = neighbors(1)
            return triad * closing / (total(closing) + SMALLEST_TOTAL) + (1 - triad) * attaching / total(attaching)

        return [
            *super()._get_laws(s),
            UpdateLaw(s.near, neighbors(where(s.other.age < 0.5, 1, 0))),  # noqa: PLR2004
            UpdateLaw(
                TopologyRequestSpec('add-edge', edges=int(self.extra), toward=weight),
                where(s.age < 0.5, 1, 0),  # noqa: PLR2004
            ),
        ]


class CopyingVertexElement(VertexElement):
    """Node of a vertex-copying network, which holds the inverse of its degree for the copying of its edges.

    Args:
        label:
            The label of the node group.

        inverse:
            The inverse of the degree, which only the neighbours of the node read.

    Examples:
        >>> from skcomplex.models.mathematics.networks import CopyingVertexElement
        >>> CopyingVertexElement('nodes').inverse
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        age: float = 0.0,
        degree: float = 0.0,
        inverse: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual, age=age, degree=degree)
        self.inverse = inverse

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the ageing, the degree, and its inverse.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [*super()._get_laws(s), UpdateLaw(s.inverse, 1 / (neighbors(1) + SMALLEST_DEGREE))]


class AttachmentElement(Element):
    """Arrival of one node per step, joined to `edges` nodes drawn by preferential attachment.

    Args:
        label:
            The label of the element.

        edges:
            The number of edges of every new node.

    Examples:
        >>> from skcomplex.models.mathematics.networks import AttachmentElement
        >>> AttachmentElement('growth', edges=3).edges
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        edges: Annotated[int, ParameterSpec(domain=NN)] = 2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.edges = edges

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the arrival of a node with its edges.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [UpdateLaw(TopologyRequestSpec('add-node', edges=int(self.edges), toward=_degree), 1)]


class CopyingElement(Element):
    """Arrival of one node per step whose edges copy the edges of random prototypes.

    Every edge of a new node joins a uniformly random node with probability `randomness`, and otherwise a random
    neighbour of a uniformly random prototype, so a node is reached with weight `randomness + (1 - randomness) Σ_j
    1 / k_j` over its neighbours `j`. Copying makes the probability of a new edge grow with the degree.

    Args:
        label:
            The label of the element.

        edges:
            The number of edges of every new node.

        randomness:
            The probability that an edge joins a random node instead of copying.

    Examples:
        >>> from skcomplex.models.mathematics.networks import CopyingElement
        >>> CopyingElement('growth', randomness=0.1).randomness
        0.1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        edges: Annotated[int, ParameterSpec(domain=NN)] = 3,
        randomness: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label, visual=visual)
        self.edges = edges
        self.randomness = randomness

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the arrival of a node with its copied edges.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        randomness = float(self.randomness)

        def weight(node: SimpleNamespace) -> object:
            """Return the chance that a copied edge or a random edge reaches a candidate node."""
            return randomness + (1 - randomness) * neighbors(node.other.inverse)

        return [UpdateLaw(TopologyRequestSpec('add-node', edges=int(self.edges), toward=weight), 1)]


def _nodes(element: type[VertexElement], capacity: int, **values: object) -> GroupElement:
    """Return the variable group on the nodes of a growing network, whose seed nodes start at age 1."""
    return GroupElement(
        'nodes',
        element=element,
        on='sites',
        variable=True,
        capacity=capacity,
        values={'age': 1.0, **values},
        visual=DEGREE_VISUAL,
    )


def _grown(nodes: GroupElement, growth: Element, space: NetworkSpace, capacity: int) -> System:
    """Return a growing network of a group on its nodes, a variable edge group, and the arrival of nodes."""
    links = GroupElement('links', element=LinkElement, on='edges', variable=True, capacity=capacity)
    return System(elements=[nodes, links, growth], space=space, time=DiscreteTime())


@register_model(
    field='mathematics/networks',
    tags=('network', 'growth', 'preferential attachment', 'scale-free'),
    references=('Barabási and Albert 1999', 'Argun et al. 2021', 'Gros 2015', 'Thurner et al. 2018'),
    defaults={'until': 300},
)
def barabasi_albert(n_nodes: int = 303, *, edges: int = 2, seed_nodes: int = 3) -> System:
    """Return Barabási-Albert growth, whose degrees follow a power law `P(k) ~ k^-3` by preferential attachment.

    The network starts complete on `seed_nodes` nodes, and every step one node joins with `edges` edges to nodes
    drawn with probability proportional to their degree, until it holds `n_nodes` nodes.

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        edges:
            The number of edges of every new node.

        seed_nodes:
            The number of nodes of the complete seed network.

    Returns:
        system:
            The model, with the groups `nodes` and `links` and the arrival `growth`.

    Examples:
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.mathematics.networks import barabasi_albert
        >>> trajectory = Simulation(barabasi_albert(13), random_state=0).run(until=10).trajectory_
        >>> int(trajectory.network(-1).sources.size)
        23
    """
    space = NetworkSpace(nx.complete_graph(seed_nodes))
    nodes = _nodes(VertexElement, n_nodes)
    return _grown(nodes, AttachmentElement('growth', edges=edges), space, edges * n_nodes + seed_nodes**2)


@register_model(
    field='mathematics/networks',
    tags=('network', 'growth', 'preferential attachment', 'clustering', 'triad formation'),
    references=('Holme and Kim 2002', 'Argun et al. 2021', 'Gros 2015', 'Thurner et al. 2018'),
    defaults={'until': 300},
)
def holme_kim(n_nodes: int = 304, *, edges: int = 3, triad: float = 0.8, seed_nodes: int = 4) -> System:
    """Return Holme-Kim growth, scale-free like Barabási-Albert growth but with clustering that stays high.

    Every step a node joins with one edge by preferential attachment, and the next step it adds `edges - 1` edges,
    each to a neighbour of its first contact with probability `triad`, closing a triangle, and by preferential
    attachment otherwise.

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        edges:
            The number of edges of every new node.

        triad:
            The probability that an added edge closes a triangle.

        seed_nodes:
            The number of nodes of the complete seed network.

    Returns:
        system:
            The model, with the groups `nodes` and `links` and the arrival `growth`.

    Examples:
        >>> from skcomplex.models.mathematics.networks import holme_kim
        >>> holme_kim(20).elements[0].element.__name__
        'TriadVertexElement'
    """
    space = NetworkSpace(nx.complete_graph(seed_nodes))
    nodes = _nodes(TriadVertexElement, n_nodes, triad=triad, extra=edges - 1)
    return _grown(nodes, AttachmentElement('growth', edges=1), space, edges * n_nodes + seed_nodes**2)


@register_model(
    field='mathematics/networks',
    tags=('network', 'growth', 'copying', 'scale-free'),
    references=('Kleinberg et al. 1999', 'Krapivsky and Redner 2005', 'Ramasco 2020'),
    defaults={'until': 300},
)
def vertex_copying(n_nodes: int = 304, *, edges: int = 3, randomness: float = 0.2, seed_nodes: int = 4) -> System:
    """Return vertex-copying growth, whose new nodes copy the edges of existing nodes and make a heavy-tailed degree.

    Every step a node joins with `edges` edges, each to a random node with probability `randomness` and otherwise to
    a random neighbour of a random prototype, which favours nodes of high degree without reading degrees.

    Args:
        n_nodes:
            The number of nodes the network holds room for.

        edges:
            The number of edges of every new node.

        randomness:
            The probability that an edge joins a random node instead of copying.

        seed_nodes:
            The number of nodes of the complete seed network.

    Returns:
        system:
            The model, with the groups `nodes` and `links` and the arrival `growth`.

    Examples:
        >>> from skcomplex.models.mathematics.networks import vertex_copying
        >>> vertex_copying(20).elements[2].randomness
        0.2
    """
    space = NetworkSpace(nx.complete_graph(seed_nodes))
    nodes = _nodes(CopyingVertexElement, n_nodes)
    growth = CopyingElement('growth', edges=edges, randomness=randomness)
    return _grown(nodes, growth, space, edges * n_nodes + seed_nodes**2)
