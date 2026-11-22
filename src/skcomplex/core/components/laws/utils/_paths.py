"""Define the reduction over the shortest paths of the active links of a group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression

from ....enums import PathReductionKind
from ....errors import ValidationError
from ...spaces import BaseNeighborhood, LayerSpec
from ._reductions import SHORTEST_PATHS, _neighborhood_symbol


def shortest_paths(
    active: object,
    over: BaseNeighborhood | LayerSpec | None = None,
    reduce: str = 'betweenness',
) -> Expression:
    """Return how central every member is on the shortest paths of active links.

    A link joins two neighbours of a group, read in both directions, and it is active where `active` is not 0, as in
    `components`, such as `shortest_paths(edges.up, reduce='closeness')` for how close every member is to the others
    over the links that are up. The graph of active links is read again at every step, so the measure follows a network
    that changes during a run, and NetworkX computes it again only when the active links changed. It gives the load on
    the nodes of a power grid, the brokers of a trade network, and the hubs of a road network.

    Read more in the user guide, under Laws.

    Args:
        active:
            The activity of one link, an expression of the symbols of `namespace.other`, `namespace`, and the edge
            between them, whose links are active where it is not 0.

        over:
            The neighbourhood, or `None` for the neighbours of the network or lattice the group lives on.

        reduce:
            What to give every member. `'betweenness'` gives the number of shortest paths between other members that
            pass through it, each path counted as its share of the shortest paths of its pair, as
            `networkx.betweenness_centrality(graph, normalized=False)` gives it. `'closeness'` gives the closeness of
            `networkx.closeness_centrality(graph)`, scaled by the share of members it reaches.

    Returns:
        reduction:
            The unevaluated reduction, one value per member.

    Raises:
        ValidationError:
            The reduction is unknown.

    See Also:
        components: Finds the clusters of the active links.
        neighbors: Reduces the values of the neighbours of every member.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import shortest_paths
        >>> shortest_paths(var('up'), reduce='closeness')
        shortest_paths(up, reduce='closeness')
    """
    if reduce not in PathReductionKind:
        error_msg = f'Unknown shortest paths reduction {reduce!r}. Use one of {", ".join(PathReductionKind)}.'
        raise ValidationError(error_msg)
    return SHORTEST_PATHS(SR(active), _neighborhood_symbol(over), tuple(PathReductionKind).index(reduce))
