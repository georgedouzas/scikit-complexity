"""Growing networks: preferential attachment, triad formation, and vertex copying."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ....core.base import BaseLaw, NumericDomain, VisualSpec, parameter, state
from ....core.components import ElementComponent, GroupComponent
from ....core.laws import Symbols, UpdateLaw, neighbors, total, where
from ....core.space import DiscreteTime, NetworkSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....formalisms.topology import add_edge, add_node

DEGREE_VISUAL = VisualSpec('sites', {'states': ['degree']})
SMALLEST_TOTAL = 1e-12
SMALLEST_DEGREE = 1e-9


def _degree(node: Symbols) -> object:
    """Return the degree of a candidate node, its weight under preferential attachment."""
    return neighbors(1)


class LinkComponent(ElementComponent):
    """Edge of a growing network, which holds no state of its own.

    Args:
        label:
            The label of the edge group.

    Examples:
        >>> from skcomplex.models.mathematics.networks import LinkComponent
        >>> LinkComponent('links').label
        'links'
    """


class VertexComponent(ElementComponent):
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
        >>> from skcomplex.models.mathematics.networks import VertexComponent
        >>> VertexComponent('nodes').age
        0.0
    """

    visual: ClassVar[VisualSpec] = DEGREE_VISUAL
    age: float = state(default=0.0, domain=NumericDomain.NONNEGATIVE)
    degree: float = state(default=0.0, domain=NumericDomain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the ageing and the degree of the node.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [UpdateLaw(s.age, s.age + 1), UpdateLaw(s.degree, neighbors(1))]


class TriadVertexComponent(VertexComponent):
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
        >>> from skcomplex.models.mathematics.networks import TriadVertexComponent
        >>> TriadVertexComponent('nodes', triad=0.5).triad
        0.5
    """

    near: float = state(default=0.0, domain=NumericDomain.NONNEGATIVE)
    triad: float = parameter(default=0.8, domain=NumericDomain.NONNEGATIVE)
    extra: int = parameter(default=1, domain=NumericDomain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the ageing, the degree, the marks next to the newest node, and the edges of triad formation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        triad = s.triad

        def weight(node: Symbols) -> object:
            """Return the mixture of triad formation and preferential attachment for a candidate node."""
            closing = neighbors(node.other.near)
            attaching = neighbors(1)
            return triad * closing / (total(closing) + SMALLEST_TOTAL) + (1 - triad) * attaching / total(attaching)

        return [
            *super().laws(s),
            UpdateLaw(s.near, neighbors(where(s.other.age < 0.5, 1, 0))),  # noqa: PLR2004
            UpdateLaw(add_edge(edges=int(self.extra), toward=weight), where(s.age < 0.5, 1, 0)),  # noqa: PLR2004
        ]


class CopyingVertexComponent(VertexComponent):
    """Node of a vertex-copying network, which holds the inverse of its degree for the copying of its edges.

    Args:
        label:
            The label of the node group.

        inverse:
            The inverse of the degree, which only the neighbours of the node read.

    Examples:
        >>> from skcomplex.models.mathematics.networks import CopyingVertexComponent
        >>> CopyingVertexComponent('nodes').inverse
        1.0
    """

    inverse: float = state(default=1.0, domain=NumericDomain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the ageing, the degree, and its inverse.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [*super().laws(s), UpdateLaw(s.inverse, 1 / (neighbors(1) + SMALLEST_DEGREE))]


class AttachmentComponent(ElementComponent):
    """Arrival of one node per step, joined to `edges` nodes drawn by preferential attachment.

    Args:
        label:
            The label of the element.

        edges:
            The number of edges of every new node.

    Examples:
        >>> from skcomplex.models.mathematics.networks import AttachmentComponent
        >>> AttachmentComponent('growth', edges=3).edges
        3
    """

    edges: int = parameter(default=2, domain=NumericDomain.NONNEGATIVE_INTEGER)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the arrival of a node with its edges.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [UpdateLaw(add_node(edges=int(self.edges), toward=_degree), 1)]


class CopyingComponent(ElementComponent):
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
        >>> from skcomplex.models.mathematics.networks import CopyingComponent
        >>> CopyingComponent('growth', randomness=0.1).randomness
        0.1
    """

    edges: int = parameter(default=3, domain=NumericDomain.NONNEGATIVE_INTEGER)
    randomness: float = parameter(default=0.2, domain=NumericDomain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the arrival of a node with its copied edges.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        randomness = float(self.randomness)

        def weight(node: Symbols) -> object:
            """Return the chance that a copied edge or a random edge reaches a candidate node."""
            return randomness + (1 - randomness) * neighbors(node.other.inverse)

        return [UpdateLaw(add_node(edges=int(self.edges), toward=weight), 1)]


def _nodes(element: type[VertexComponent], capacity: int, **values: object) -> GroupComponent:
    """Return the variable group on the nodes of a growing network, whose seed nodes start at age 1."""
    return GroupComponent(
        'nodes',
        element=element,
        on='sites',
        variable=True,
        capacity=capacity,
        values={'age': 1.0, **values},
        visual=DEGREE_VISUAL,
    )


def _grown(nodes: GroupComponent, growth: ElementComponent, space: NetworkSpace, capacity: int) -> System:
    """Return a growing network of a group on its nodes, a variable edge group, and the arrival of nodes."""
    links = GroupComponent('links', element=LinkComponent, on='edges', variable=True, capacity=capacity)
    return System(components=[nodes, links, growth], space=space, time=DiscreteTime())


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
    space = NetworkSpace.complete(seed_nodes)
    nodes = _nodes(VertexComponent, n_nodes)
    return _grown(nodes, AttachmentComponent('growth', edges=edges), space, edges * n_nodes + seed_nodes**2)


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
        >>> holme_kim(20).components[0].element.__name__
        'TriadVertexComponent'
    """
    space = NetworkSpace.complete(seed_nodes)
    nodes = _nodes(TriadVertexComponent, n_nodes, triad=triad, extra=edges - 1)
    return _grown(nodes, AttachmentComponent('growth', edges=1), space, edges * n_nodes + seed_nodes**2)


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
        >>> vertex_copying(20).components[2].randomness
        0.2
    """
    space = NetworkSpace.complete(seed_nodes)
    nodes = _nodes(CopyingVertexComponent, n_nodes)
    growth = CopyingComponent('growth', edges=edges, randomness=randomness)
    return _grown(nodes, growth, space, edges * n_nodes + seed_nodes**2)
