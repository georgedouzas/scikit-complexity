"""Define reductions of expressions over the neighbours of a member in a network, a lattice, or a continuous space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib
from dataclasses import dataclass

from sage.all__sagemath_symbolics import SR
from sage.symbolic.function import BuiltinFunction

from ..base import Expression, SymbolicVector, ValidationError, symbol, vector
from ..space import Adjacency, Cone, Layer, Memory, Nearest, Neighborhood, Radius
from ._lattices import neighbor

NEIGHBOUR_REDUCTIONS = ('sum', 'mean', 'count', 'max', 'min', 'normalized-mean')
NEIGHBOURHOOD_TYPES = (Adjacency, Neighborhood, Memory, Radius, Nearest, Cone)
NEIGHBOURHOODS: dict[str, object] = {}

type NeighbourhoodSpec = Adjacency | Neighborhood | Memory | Radius | Nearest | Cone


def _over_symbol(over: object) -> Expression:
    """Return the symbol that stands for a neighbourhood, registered by its value, or 0 for the space's own."""
    if over is None:
        return SR(0)
    if isinstance(over, Layer):
        over = Adjacency(layer=over.label)
    if not isinstance(over, NEIGHBOURHOOD_TYPES):
        names = ', '.join(item.__name__ for item in (*NEIGHBOURHOOD_TYPES, Layer))
        error_msg = f'A neighbourhood reduction reduces over None or one of {names}, not {over!r}.'
        raise ValidationError(error_msg)
    digest = hashlib.sha256(repr(over).encode()).hexdigest()[:16]
    name = f'_over_{digest}'
    NEIGHBOURHOODS[name] = over
    return symbol(name)


class _NeighbourReduction(BuiltinFunction):
    """Reduction over the neighbours of a member, `neighbors(argument, weight, over, kind)`, kept unevaluated."""

    def __init__(self) -> None:
        super().__init__('neighbors', nargs=4)

    def _eval_(self, *operands: Expression) -> None:
        """Return `None`, so the call stays unevaluated."""
        del operands

    def _print_(self, *operands: Expression) -> str:
        """Return the call with its keywords, leaving out the space's own neighbourhood and the unit weight."""
        argument, weight, over, kind = operands
        parts = [str(argument)]
        if not _is_number(over, 0):
            parts.append(f'over={NEIGHBOURHOODS.get(str(over), over)!r}')
        parts.append(f'reduce={NEIGHBOUR_REDUCTIONS[int(SR(kind))]!r}')
        if not _is_number(weight, 1):
            parts.append(f'weight={weight}')
        return f'neighbors({", ".join(parts)})'


def _is_number(item: Expression, value: int) -> bool:
    """Check whether an operand is a given number, the placeholder of an absent neighbourhood or weight."""
    item = SR(item)
    return bool(item.is_numeric() and item == value)


REDUCE = _NeighbourReduction()


def neighbors(
    expression: object,
    over: NeighbourhoodSpec | Layer | None = None,
    reduce: str | None = None,
    weight: object = None,
) -> Expression | SymbolicVector:
    """Return the reduction of an expression over the neighbours of every member of a group.

    The expression is written for one neighbour: the namespace `s.other` holds the symbols of the neighbour, and `s`
    those of the member that reads the reduction, so `sin(s.other.theta - s.theta)` is the phase difference to one
    neighbour. The neighbourhood is the graph of a network space by default, `Adjacency()`, a layer of the network,
    `Adjacency(layer=label)` or the `Layer` itself, the neighbourhood of a lattice space, or in a continuous space the
    members within a distance, `Radius(r)`, the nearest members, `Nearest(k)`, or the members within a distance
    inside a cone around a heading, `Cone(r, angle)`. A reduction over no neighbours is 0.

    The reductions with the weight `w_j` of every neighbour `j` of the value `f_j` are:

    - `'sum'`, the sum `sum_j w_j f_j`;
    - `'mean'`, the sum divided by the number of neighbours;
    - `'count'`, the weight of the neighbours whose value is not zero, `sum_j w_j [f_j != 0]`;
    - `'max'` and `'min'`, the largest and smallest `w_j f_j`;
    - `'normalized-mean'`, the weighted mean `sum_j w_j f_j / sum_j w_j`.

    A lattice neighbourhood without a reduction keeps the lattice case of local energies: the values of a lattice state
    at every offset, in neighbourhood order.

    Args:
        expression:
            The value of one neighbour, an expression of the symbols of `s.other` and `s`, or a vector of them.

        over:
            The neighbourhood, or `None` for the neighbours of the network or lattice the group lives on.

        reduce:
            `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, or `'normalized-mean'`, or `None` for the sum, or for the
            values at every offset of a lattice neighbourhood.

        weight:
            The weight of one neighbour, an expression of the same symbols, or `None` for 1. On a network it may read
            the states of a group on the edges of the layer, through a role of an interaction bound to the group, as
            the state of the edge that joins the member to the neighbour.

    Returns:
        reduction:
            The unevaluated reduction, one per component of a vector expression, or the values of a lattice state at
            the offsets of a lattice neighbourhood.

    Raises:
        ValidationError:
            The reduction or the neighbourhood is unknown.

    Examples:
        The differences to the neighbours on a path graph relax every member towards them:

        >>> import networkx as nx
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.core.components import Element
        >>> from skcomplex.core.laws import Rate, neighbors
        >>> from skcomplex.core.space import NetworkSpace
        >>> from skcomplex.core.system import System
        >>> Relaxing = Element.define(
        ...     'Relaxing', states={'u': 0.0}, laws=lambda s: [Rate(s.u, neighbors(s.other.u - s.u))],
        ... )
        >>> system = System(
        ...     components=[Relaxing.group('chain', 3, u=[0.0, 1.0, 4.0])], space=NetworkSpace(nx.path_graph(3)),
        ... )
        >>> problem = compile_system(system)
        >>> problem.drift(0.0, problem.initial_state, problem.initial_parameters).tolist()
        [1.0, 2.0, -3.0]

        A reduction stays symbolic until the compiler lowers it:

        >>> from sage.all__sagemath_symbolics import sin
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.space import Neighborhood, Radius
        >>> neighbors(sin(symbol('theta_other')), over=Radius(1.0), reduce='mean')
        neighbors(sin(theta_other), over=Radius(radius=1.0, center=False), reduce='mean')

        A lattice neighbourhood without a reduction gives the values at its offsets:

        >>> spin = symbol('spin')
        >>> values = neighbors(spin, Neighborhood.von_neumann())
        >>> len(values), values[0]
        (4, neighbor(spin, 0))
    """
    if reduce is None and isinstance(over, Neighborhood | Memory):
        return vector(neighbor(expression, SR(index)) for index in range(len(over)))
    kind = 'sum' if reduce is None else reduce
    if kind not in NEIGHBOUR_REDUCTIONS:
        error_msg = f'Unknown neighbourhood reduction {kind!r}. Use one of {", ".join(NEIGHBOUR_REDUCTIONS)}.'
        raise ValidationError(error_msg)
    over_symbol = _over_symbol(over)
    code = SR(NEIGHBOUR_REDUCTIONS.index(kind))
    weighted = SR(1) if weight is None else SR(weight)
    if isinstance(expression, SymbolicVector):
        return vector(REDUCE(SR(item), weighted, over_symbol, code) for item in expression)
    return REDUCE(SR(expression), weighted, over_symbol, code)


@dataclass(frozen=True)
class NeighbourReduction:
    """Neighbourhood reduction found in an expression, with its kind, argument, weight, and neighbourhood.

    Args:
        call:
            The unevaluated call.

        kind:
            One of `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, and `'normalized-mean'`.

        argument:
            The value of one neighbour.

        weight:
            The weight of one neighbour, or `None` for 1.

        over:
            The neighbourhood, or `None` for the neighbours of the space.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import neighbors, neighbourhoods_in
        >>> neighbourhoods_in(1 + neighbors(symbol('x'), reduce='max'))[0].kind
        'max'
    """

    call: Expression
    kind: str
    argument: Expression
    weight: Expression | None
    over: NeighbourhoodSpec | None


def _reduction_of(call: Expression) -> NeighbourReduction:
    """Return the parts of a neighbourhood reduction call."""
    argument, weight, over, kind = call.operands()
    return NeighbourReduction(
        call=call,
        kind=NEIGHBOUR_REDUCTIONS[int(SR(kind))],
        argument=argument,
        weight=None if _is_number(weight, 1) else weight,
        over=None if _is_number(over, 0) else NEIGHBOURHOODS[str(over)],  # type: ignore[arg-type]
    )


def neighbourhoods_in(expression: object) -> list[NeighbourReduction]:
    """Return the outermost neighbourhood reductions of an expression.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        reductions:
            The neighbourhood reductions, each listed once.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import neighbors, neighbourhoods_in
        >>> [item.kind for item in neighbourhoods_in(neighbors(symbol('x'), reduce='mean'))]
        ['mean']
    """
    if not isinstance(expression, Expression | SymbolicVector | int | float):
        return []
    found: dict[str, NeighbourReduction] = {}
    stack = [SR(item) for item in (list(expression) if isinstance(expression, SymbolicVector) else [expression])]
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        if item.operator() == REDUCE:
            found.setdefault(str(item), _reduction_of(item))
            continue
        stack.extend(item.operands())
    return list(found.values())
