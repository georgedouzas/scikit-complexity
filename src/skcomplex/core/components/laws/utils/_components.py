"""Define the reduction over the connected components of the active links."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression

from ....enums import ComponentReductionKind
from ....errors import ValidationError
from ...spaces import BaseNeighborhood, LayerSpec
from ._reductions import COMPONENTS, _neighborhood_symbol


def components(
    active: object,
    over: BaseNeighborhood | LayerSpec | None = None,
    reduce: str = 'label',
) -> Expression:
    """Return the clusters of members joined by active links, as a value per member.

    A link joins two neighbours of a group, and it is active where `active` is not 0, such as
    `components(edges.open > 0, reduce='size')` for the size of the cluster of open links each member is in. Inside
    `active`, `namespace.other` holds the values of the neighbour, `namespace` those of the member, and a group on the
    edges those of the edge. Two members share a cluster when a path of active links joins them, read in both
    directions, and a member without active links is a cluster of its own. It finds the spanning cluster of a porous
    rock, the outbreak cluster of an epidemic, and the groups of firms that a default can reach.

    Read more in the user guide, under Laws.

    Args:
        active:
            The value of one link, an expression of the values of `namespace.other`, `namespace`, and the edge between
            them, whose links are active where it is not 0.

        over:
            The neighbourhood, or `None` for the neighbours of the network or lattice the group lives on.

        reduce:
            What to give every member. `'label'` gives the smallest index of a member of its cluster, `'size'` the
            number of members of its cluster, `'largest'` the number of members of the largest cluster, and `'giant'`
            gives 1 in the largest cluster and 0 elsewhere, taking the smallest label among clusters of equal size.

    Returns:
        reduction:
            The unevaluated reduction, one value per member.

    Raises:
        ValidationError:
            The reduction or the neighbourhood is unknown.

    See Also:
        neighbors: Reduces the values of the neighbours of every member.
        shortest_paths: Measures the shortest paths of the active links.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import components
        >>> components(var('open') > 0, reduce='size')
        components(open > 0, reduce='size')
    """
    if reduce not in ComponentReductionKind:
        error_msg = f'Unknown components reduction {reduce!r}. Use one of {", ".join(ComponentReductionKind)}.'
        raise ValidationError(error_msg)
    return COMPONENTS(SR(active), _neighborhood_symbol(over), tuple(ComponentReductionKind).index(reduce))
