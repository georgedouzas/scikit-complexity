"""Define the reduction over the connected components of the active links.

The links are those between the members of a group.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression

from ....errors import ValidationError
from ....specs import LayerSpec
from ...spaces import BaseNeighborhood
from ._reductions import COMPONENT_REDUCTIONS, COMPONENTS, _neighborhood_symbol


def components(
    active: object,
    over: BaseNeighborhood | LayerSpec | None = None,
    reduce: str = 'label',
) -> Expression:
    """Return a reduction over the connected components of the active links.

    The links are those between the members of a group. The links are the pairs of neighbours of the neighbourhood, the
    graph of a network space by default, and a link is active when `active`, written for one link as in `neighbors`, is
    not 0. Inside it `namespace.other` holds the symbols of the neighbour, `namespace` those of the member, and the
    states of a group on the edges of the network those of the edge that joins them, so a link is active by a condition,
    such as `edges.state > 0`, or by an edge weight that is not 0. Two members are in one component when a path of
    active links joins them, read without direction on a directed network, and a member without active links is a
    component of its own.

    The reductions are `'label'`, the smallest index of a member of the component of every member, `'size'`, the
    number of members of the component of every member, `'largest'`, the number of members of the largest component,
    the same for every member, and `'giant'`, 1 for the members of the largest component and 0 for the others, the
    component of the smallest label among components of equal size.

    Args:
        active:
            The activity of one link, an expression of the symbols of `namespace.other`, `namespace`, and the edge
            between them, whose links are active where it is not 0.

        over:
            The neighbourhood, or `None` for the neighbours of the network or lattice the group lives on.

        reduce:
            `'label'`, `'size'`, `'largest'`, or `'giant'`.

    Returns:
        reduction:
            The unevaluated reduction, one value per member.

    Raises:
        ValidationError:
            The reduction or the neighbourhood is unknown.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import components, neighborhoods_in
        >>> components(var('open') > 0, reduce='size')
        components(open > 0, reduce='size')
        >>> [item.kind for item in neighborhoods_in(1 + components(var('open'), reduce='largest'))]
        ['component-largest']
    """
    if reduce not in COMPONENT_REDUCTIONS:
        error_msg = f'Unknown components reduction {reduce!r}. Use one of {", ".join(COMPONENT_REDUCTIONS)}.'
        raise ValidationError(error_msg)
    return COMPONENTS(SR(active), _neighborhood_symbol(over), COMPONENT_REDUCTIONS.index(reduce))
