"""Define the reduction over the shortest paths of the active links of a group."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression

from ....enums import PathReductionKind
from ....errors import ValidationError
from ....specs import LayerSpec
from ...spaces import BaseNeighborhood
from ._reductions import SHORTEST_PATHS, _neighborhood_symbol


def shortest_paths(
    active: object,
    over: BaseNeighborhood | LayerSpec | None = None,
    reduce: str = 'betweenness',
) -> Expression:
    """Return a reduction over the shortest paths of the active links.

    The links are those between the members of a group. The links are the pairs of neighbours of the neighbourhood, the
    graph of a network space by default, read without direction, and a link is active when `active`, written for one
    link as in `components`, is not 0. The reduction reads the graph of the active links at every evaluation, so it
    follows a network whose topology changes during a run, and NetworkX computes it again only when the active links
    changed since the last evaluation.

    The reductions are `'betweenness'`, the number of shortest paths between other pairs of members that pass through
    every member, each path between a pair counted as a share of the shortest paths of the pair, as
    `networkx.betweenness_centrality(graph, normalized=False)` gives it, the load of the cascades of Motter and Lai,
    and `'closeness'`, the closeness of every member as `networkx.closeness_centrality(graph)` gives it, scaled by the
    share of members it reaches.

    Args:
        active:
            The activity of one link, an expression of the symbols of `namespace.other`, `namespace`, and the edge
            between them, whose links are active where it is not 0.

        over:
            The neighbourhood, or `None` for the neighbours of the network or lattice the group lives on.

        reduce:
            `'betweenness'` or `'closeness'`.

    Returns:
        reduction:
            The unevaluated reduction, one value per member.

    Raises:
        ValidationError:
            The reduction is unknown.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import shortest_paths
        >>> from skcomplex.core.components.laws.utils._reductions import neighborhoods_in
        >>> shortest_paths(1)
        shortest_paths(1, reduce='betweenness')
        >>> [item.kind for item in neighborhoods_in(var('c') - shortest_paths(var('up'), reduce='closeness'))]
        ['path-closeness']
    """
    if reduce not in PathReductionKind:
        error_msg = f'Unknown shortest paths reduction {reduce!r}. Use one of {", ".join(PathReductionKind)}.'
        raise ValidationError(error_msg)
    return SHORTEST_PATHS(SR(active), _neighborhood_symbol(over), tuple(PathReductionKind).index(reduce))
