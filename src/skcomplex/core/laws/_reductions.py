"""Define the symbolic reductions laws use over the members of a group, a set of elements, or the neighbours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import SR
from sage.symbolic.function import BuiltinFunction

from ..base import Expression, SymbolicVector, ValidationError, symbol, vector
from ..space import Adjacency, Cone, Layer, Memory, Nearest, Neighborhood, Radius
from ._utils import Symbols

RELATION_MARKERS = {
    operator.lt: 'skcomplex_lt',
    operator.le: 'skcomplex_le',
    operator.gt: 'skcomplex_gt',
    operator.ge: 'skcomplex_ge',
    operator.eq: 'skcomplex_eq',
    operator.ne: 'skcomplex_ne',
}
WEIGHTS: dict[str, Any] = {}

type Weights = np.ndarray | scipy.sparse.sparray | scipy.sparse.spmatrix


def _is_trivial(item: Expression, value: int) -> bool:
    """Check whether an operand holds a placeholder number, the absent condition, weights, or group."""
    item = SR(item)
    return bool(item.is_numeric() and item == value)


def _rebuild(item: Expression, replace: Any) -> Expression:  # noqa: ANN401
    """Return an expression rebuilt bottom up, with every subexpression passed through a replacement function."""
    item = SR(item)
    if item.operator() is None:
        return item
    operands = [_rebuild(operand, replace) for operand in item.operands()]
    return replace(item, operands)


class _Marker(BuiltinFunction):
    """Relation written as a function, which carries a condition through Maxima, `skcomplex_gt(lhs, rhs)`."""

    def __init__(self, name: str, relation: Any) -> None:  # noqa: ANN401
        super().__init__(name, nargs=2)
        self.relation = relation


MARKERS = {relation: _Marker(name, relation) for relation, name in RELATION_MARKERS.items()}
MARKED = {marker.name(): marker.relation for marker in MARKERS.values()}


def _marked(condition: Expression) -> Expression:
    """Return a condition with every relation replaced by its marker function."""

    def replace(item: Expression, operands: list[Expression]) -> Expression:
        relation = item.operator()
        if relation in MARKERS:
            return MARKERS[relation](*operands)
        return relation(*operands)

    return _rebuild(condition, replace)


def _restored(condition: Expression) -> Expression:
    """Return a condition with every marker function replaced by its relation."""

    def replace(item: Expression, operands: list[Expression]) -> Expression:
        name = getattr(item.operator(), 'name', lambda: '')()
        if name in MARKED:
            return MARKED[name](*operands)
        return item.operator()(*operands)

    return _rebuild(condition, replace)


def _has_markers(condition: Expression) -> bool:
    """Check whether a condition holds a marker function."""
    stack = [SR(condition)]
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        if getattr(item.operator(), 'name', lambda: '')() in MARKED:
            return True
        stack.extend(item.operands())
    return False


class _Reduction(BuiltinFunction):
    """Reduction over the members of a group, `name(argument, condition, weights, over)`, kept unevaluated."""

    def __init__(self, name: str, kind: str) -> None:
        super().__init__(name, nargs=4)
        self.kind = kind

    def _eval_(self, *operands: Expression) -> Expression | None:
        """Return the call with relations restored after Maxima, or `None` to stay unevaluated."""
        argument, condition, weights, over = operands
        if _has_markers(condition):
            return self(argument, _restored(condition), weights, over)
        return None

    def _print_(self, *operands: Expression) -> str:
        """Return the call with its keywords, leaving out the absent condition, weights, and group."""
        argument, condition, weights, over = operands
        parts = [] if self.kind == 'count' else [str(argument)]
        if not _is_trivial(over, 0):
            parts.append(f'over={over}')
        if not _is_trivial(condition, 1):
            parts.append(f'where={condition}')
        if not _is_trivial(weights, 0):
            parts.append(f'weights={weights}')
        return f'{self.name()}({", ".join(parts)})'

    def _maxima_init_evaled_(self, *operands: Expression) -> str:
        """Return the call in Maxima's syntax, with relations of the condition written as marker functions."""
        argument, condition, weights, over = operands
        codes = [SR(item)._maxima_init_() for item in (argument, _marked(SR(condition)), weights, over)]
        return f'{self.name()}({",".join(codes)})'


FUNCTIONS = {
    'total': _Reduction('total', 'sum'),
    'mean': _Reduction('mean', 'mean'),
    'maximum': _Reduction('maximum', 'max'),
    'minimum': _Reduction('minimum', 'min'),
    'variance': _Reduction('variance', 'var'),
    'count': _Reduction('count', 'count'),
}
KINDS = {function: function.kind for function in FUNCTIONS.values()}


def _over_symbol(over: object) -> Expression:
    """Return a symbol that names the group of a reduction, from a namespace or a symbol, or 0 for none."""
    if over is None:
        return SR(0)
    if isinstance(over, Symbols):
        for name, value in vars(over).items():
            if name == 't':
                continue
            if isinstance(value, SymbolicVector):
                return SR(next(iter(value)))
            if isinstance(value, Expression) and value.is_symbol():
                return value
    if isinstance(over, Expression) and over.is_symbol():
        return over
    error_msg = (
        f'A reduction takes the group to reduce over as a namespace such as `s.neurons`, or as one of its symbols, '
        f'not {over!r}.'
    )
    raise ValidationError(error_msg)


def _weights_symbol(weights: object) -> Expression:
    """Return the symbol that stands for a constant weight matrix, or 0 for none."""
    if weights is None:
        return SR(0)
    matrix = scipy.sparse.csr_array(weights) if scipy.sparse.issparse(weights) else np.asarray(weights, dtype=float)
    if matrix.ndim not in {1, 2}:
        error_msg = (
            'The weights of a reduction are a matrix with one row per reading member, or one row that every reading '
            f'member shares, not of shape {matrix.shape}.'
        )
        raise ValidationError(error_msg)
    if scipy.sparse.issparse(matrix):
        csr = scipy.sparse.csr_array(matrix, dtype=float)
        parts = [np.asarray(csr.shape), csr.indptr, csr.indices, csr.data]
    else:
        parts = [np.asarray(matrix.shape), np.ascontiguousarray(matrix)]
    digest = hashlib.sha256(b''.join(np.asarray(part).tobytes() for part in parts)).hexdigest()[:16]
    name = f'_weights_{digest}'
    WEIGHTS[name] = matrix
    return symbol(name)


def _call(name: str, expression: object, over: object, condition: object, weights: object) -> Expression:
    """Return the unevaluated call of a reduction."""
    function = FUNCTIONS[name]
    kept = SR(1) if condition is None else SR(condition)
    return function(SR(expression), kept, _weights_symbol(weights), _over_symbol(over))


def _reduction_of(call: Expression) -> 'Reduction':
    """Return the parts of a reduction call."""
    argument, condition, weights, over = call.operands()
    return Reduction(
        call=call,
        kind=KINDS[call.operator()],
        argument=argument,
        condition=None if _is_trivial(condition, 1) else condition,
        weights=None if _is_trivial(weights, 0) else WEIGHTS[str(weights)],
        over=None if _is_trivial(over, 0) else over,
    )


def total(
    expression: object,
    over: object = None,
    where: object = None,
    weights: object = None,
) -> Expression:
    """Return the sum of an expression over the members of a group, `sum_j a_j`, or `sum_j W_ij a_j` with weights.

    Inside the reduction, the symbols of a member stand for the reduced member, and outside it for the reading member.
    The group is the one whose symbols the expression reads.

    Args:
        expression:
            The expression of one member, such as `s.x` or `cos(s.neurons.theta)`.

        over:
            The namespace of the group to reduce over, such as `s.neurons`, or `None` for the group the expression
            reads.

        where:
            A condition on the reduced member that selects the members, or `None` for all members.

        weights:
            A constant matrix, a NumPy array or a SciPy sparse matrix, with one row per reading member and one column
            per reduced member, a vector with one weight per reduced member that every reading member shares, or
            `None`.

    Returns:
        reduction:
            The unevaluated reduction, which every backend lowers.

    Raises:
        ValidationError:
            The group is not a namespace or a symbol, or the weights are not a matrix.

    Examples:
        >>> from sage.all__sagemath_symbolics import tanh
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import total
        >>> x = symbol('x')
        >>> -x + tanh(total(x))
        -x + tanh(total(x))
        >>> total(x, where=x > 0.5)
        total(x, where=x > 0.5)
    """
    return _call('total', expression, over, where, weights)


def mean(
    expression: object,
    over: object = None,
    where: object = None,
    weights: object = None,
) -> Expression:
    """Return the mean of an expression over the selected members of a group, the total divided by their count.

    Args:
        expression:
            The expression of one member.

        over:
            The namespace of the group to reduce over, or `None` for the group the expression reads.

        where:
            A condition on the reduced member that selects the members, or `None` for all members.

        weights:
            A constant matrix with one row per reading member and one column per reduced member, or `None`.

    Returns:
        reduction:
            The unevaluated reduction, 0 when no member is selected.

    Raises:
        ValidationError:
            The group is not a namespace or a symbol, or the weights are not a matrix.

    Examples:
        >>> from sage.all__sagemath_symbolics import cos
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import mean
        >>> theta = symbol('theta')
        >>> mean(cos(theta))
        mean(cos(theta))
    """
    return _call('mean', expression, over, where, weights)


def maximum(expression: object, over: object = None, where: object = None) -> Expression:
    """Return the largest value of an expression over the selected members of a group.

    Args:
        expression:
            The expression of one member.

        over:
            The namespace of the group to reduce over, or `None` for the group the expression reads.

        where:
            A condition on the reduced member that selects the members, or `None` for all members.

    Returns:
        reduction:
            The unevaluated reduction, 0 when no member is selected.

    Raises:
        ValidationError:
            The group is not a namespace or a symbol.

    Examples:
        >>> from sage.all__sagemath_symbolics import heaviside
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import maximum
        >>> x, theta_z = symbol('x'), symbol('theta_z')
        >>> heaviside(maximum(x) - theta_z)
        heaviside(-theta_z + maximum(x))
    """
    return _call('maximum', expression, over, where, None)


def minimum(expression: object, over: object = None, where: object = None) -> Expression:
    """Return the smallest value of an expression over the selected members of a group.

    Args:
        expression:
            The expression of one member.

        over:
            The namespace of the group to reduce over, or `None` for the group the expression reads.

        where:
            A condition on the reduced member that selects the members, or `None` for all members.

    Returns:
        reduction:
            The unevaluated reduction, 0 when no member is selected.

    Raises:
        ValidationError:
            The group is not a namespace or a symbol.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import minimum
        >>> minimum(symbol('x'))
        minimum(x)
    """
    return _call('minimum', expression, over, where, None)


def variance(expression: object, over: object = None, where: object = None) -> Expression:
    """Return the population variance of an expression over the selected members of a group.

    Args:
        expression:
            The expression of one member.

        over:
            The namespace of the group to reduce over, or `None` for the group the expression reads.

        where:
            A condition on the reduced member that selects the members, or `None` for all members.

    Returns:
        reduction:
            The unevaluated reduction, 0 when no member is selected.

    Raises:
        ValidationError:
            The group is not a namespace or a symbol.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import variance
        >>> variance(symbol('x'))
        variance(x)
    """
    return _call('variance', expression, over, where, None)


def count(over: object = None, where: object = None) -> Expression:
    """Return the number of members of a group that a condition selects, or of all members.

    Args:
        over:
            The namespace of the group to count, or `None` for the group the condition reads, or else the group of the
            law.

        where:
            A condition on the counted member, or `None` for all members.

    Returns:
        reduction:
            The unevaluated count.

    Raises:
        ValidationError:
            The group is not a namespace or a symbol.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import count
        >>> x = symbol('x')
        >>> count(where=x > 0.5)
        count(where=x > 0.5)
    """
    return _call('count', 1, over, where, None)


@dataclass(frozen=True)
class Reduction:
    """Reduction call found in an expression, with its kind, argument, condition, weights, and group.

    Args:
        call:
            The unevaluated call.

        kind:
            The key of the reduction in the table of `reduce_values`: `'sum'`, `'mean'`, `'max'`, `'min'`, `'var'`,
            or `'count'`.

        argument:
            The expression of one reduced member, 1 for a count.

        condition:
            The condition that selects the members, or `None` for all members.

        weights:
            The constant weight matrix, or `None`.

        over:
            A symbol of the group to reduce over, or `None` for the group the argument reads.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import reductions_in, total
        >>> x = symbol('x')
        >>> reductions_in(1 + total(x**2))[0].kind
        'sum'
    """

    call: Expression
    kind: str
    argument: Expression
    condition: Expression | None
    weights: Any
    over: Expression | None


def reductions_in(expression: object) -> list[Reduction]:
    """Return the outermost reduction calls of an expression, which may hold further reductions in their arguments.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        reductions:
            The reductions, each listed once.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import mean, reductions_in, total
        >>> x = symbol('x')
        >>> [item.kind for item in reductions_in(total(x * mean(x)))]
        ['sum']
    """
    found: dict[str, Reduction] = {}
    stack = [SR(item) for item in (list(expression) if isinstance(expression, SymbolicVector) else [expression])]
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        if item.operator() in KINDS:
            found.setdefault(str(item), _reduction_of(item))
            continue
        stack.extend(item.operands())
    return list(found.values())


def has_reductions(expression: object) -> bool:
    """Check whether an expression holds a reduction call.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        found:
            Whether the expression holds a reduction.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import count, has_reductions
        >>> has_reductions(symbol('x') + count())
        True
    """
    if not isinstance(expression, Expression | SymbolicVector | int | float):
        return False
    return bool(reductions_in(expression))


NEIGHBOR_REDUCTIONS = ('sum', 'mean', 'count', 'max', 'min', 'normalized-mean')
NEIGHBORHOOD_TYPES = (Adjacency, Neighborhood, Memory, Radius, Nearest, Cone)
NEIGHBORHOODS: dict[str, object] = {}

type NeighborhoodSpec = Adjacency | Neighborhood | Memory | Radius | Nearest | Cone


def _neighbor_over_symbol(over: object) -> Expression:
    """Return the symbol that stands for a neighbourhood, registered by its value, or 0 for the space's own."""
    if over is None:
        return SR(0)
    if isinstance(over, Layer):
        over = Adjacency(layer=over.label)
    if not isinstance(over, NEIGHBORHOOD_TYPES):
        names = ', '.join(item.__name__ for item in (*NEIGHBORHOOD_TYPES, Layer))
        error_msg = f'A neighbourhood reduction reduces over None or one of {names}, not {over!r}.'
        raise ValidationError(error_msg)
    digest = hashlib.sha256(repr(over).encode()).hexdigest()[:16]
    name = f'_over_{digest}'
    NEIGHBORHOODS[name] = over
    return symbol(name)


class _NeighborReduction(BuiltinFunction):
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
            parts.append(f'over={NEIGHBORHOODS.get(str(over), over)!r}')
        parts.append(f'reduce={NEIGHBOR_REDUCTIONS[int(SR(kind))]!r}')
        if not _is_number(weight, 1):
            parts.append(f'weight={weight}')
        return f'neighbors({", ".join(parts)})'


def _is_number(item: Expression, value: int) -> bool:
    """Check whether an operand is a given number, the placeholder of an absent neighbourhood or weight."""
    item = SR(item)
    return bool(item.is_numeric() and item == value)


def _neighbor_reduction_of(call: Expression) -> 'NeighborReduction':
    """Return the parts of a neighbourhood reduction call."""
    argument, weight, over, kind = call.operands()
    return NeighborReduction(
        call=call,
        kind=NEIGHBOR_REDUCTIONS[int(SR(kind))],
        argument=argument,
        weight=None if _is_number(weight, 1) else weight,
        over=None if _is_number(over, 0) else NEIGHBORHOODS[str(over)],  # type: ignore[arg-type]
    )


class _Neighbor(BuiltinFunction):
    """Value of a lattice state at the `k`-th offset of a neighbourhood, `neighbor(x, k)`."""

    def __init__(self) -> None:
        super().__init__('neighbor', nargs=2)


REDUCE = _NeighborReduction()
neighbor = _Neighbor()


def neighbors(
    expression: object,
    over: NeighborhoodSpec | Layer | None = None,
    reduce: str = 'sum',
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

    On a lattice neighbourhood, `reduce='each'` keeps the lattice case of local energies: the values of a lattice state
    at every offset, in neighbourhood order, without reducing them.

    Args:
        expression:
            The value of one neighbour, an expression of the symbols of `s.other` and `s`, or a vector of them.

        over:
            The neighbourhood, or `None` for the neighbours of the network or lattice the group lives on.

        reduce:
            `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, or `'normalized-mean'`, or `'each'` for the values at every
            offset of a lattice neighbourhood or a memory.

        weight:
            The weight of one neighbour, an expression of the same symbols, or `None` for 1. On a network it may read
            the states of a group on the edges of the layer, through a role of an interaction bound to the group, as
            the state of the edge that joins the member to the neighbour.

    Returns:
        reduction:
            The unevaluated reduction, one per component of a vector expression, or with `reduce='each'` the values of
            a lattice state at the offsets of a lattice neighbourhood.

    Raises:
        ValidationError:
            The reduction or the neighbourhood is unknown.

    Examples:
        The differences to the neighbours on a path graph relax every member towards them:

        >>> import networkx as nx
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.core.components import Element
        >>> from skcomplex.core.laws import RateLaw, neighbors
        >>> from skcomplex.core.space import NetworkSpace
        >>> from skcomplex.core.system import System
        >>> RelaxingElement = Element.define(
        ...     'RelaxingElement', states={'u': 0.0}, laws=lambda s: [RateLaw(s.u, neighbors(s.other.u - s.u))],
        ... )
        >>> system = System(
        ...     components=[RelaxingElement.group('chain', 3, u=[0.0, 1.0, 4.0])],
        ...     space=NetworkSpace(nx.path_graph(3)),
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

        A lattice neighbourhood gives the values at its offsets with `reduce='each'`:

        >>> spin = symbol('spin')
        >>> values = neighbors(spin, Neighborhood.von_neumann(), reduce='each')
        >>> len(values), values[0]
        (4, neighbor(spin, 0))
    """
    kind = reduce
    if kind == 'each':
        if not isinstance(over, Neighborhood | Memory):
            error_msg = f"The reduction 'each' reads the offsets of a Neighborhood or a Memory, not {over!r}."
            raise ValidationError(error_msg)
        return vector(
            neighbor(expression, SR(index))
            for index in range(len(over.offsets if isinstance(over, Neighborhood) else over.elements))
        )
    if kind not in NEIGHBOR_REDUCTIONS:
        error_msg = f'Unknown neighbourhood reduction {kind!r}. Use one of {", ".join((*NEIGHBOR_REDUCTIONS, "each"))}.'
        raise ValidationError(error_msg)
    over_symbol = _neighbor_over_symbol(over)
    code = SR(NEIGHBOR_REDUCTIONS.index(kind))
    weighted = SR(1) if weight is None else SR(weight)
    if isinstance(expression, SymbolicVector):
        return vector(REDUCE(SR(item), weighted, over_symbol, code) for item in expression)
    return REDUCE(SR(expression), weighted, over_symbol, code)


@dataclass(frozen=True)
class NeighborReduction:
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
        >>> from skcomplex.core.laws import neighbors, neighborhoods_in
        >>> neighborhoods_in(1 + neighbors(symbol('x'), reduce='max'))[0].kind
        'max'
    """

    call: Expression
    kind: str
    argument: Expression
    weight: Expression | None
    over: NeighborhoodSpec | None


def neighborhoods_in(expression: object) -> list[NeighborReduction]:
    """Return the outermost neighbourhood reductions of an expression.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        reductions:
            The neighbourhood reductions, each listed once.

    Examples:
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.laws import neighbors, neighborhoods_in
        >>> [item.kind for item in neighborhoods_in(neighbors(symbol('x'), reduce='mean'))]
        ['mean']
    """
    if not isinstance(expression, Expression | SymbolicVector | int | float):
        return []
    found: dict[str, NeighborReduction] = {}
    stack = [SR(item) for item in (list(expression) if isinstance(expression, SymbolicVector) else [expression])]
    while stack:
        item = stack.pop()
        if item.operator() is None:
            continue
        if item.operator() == REDUCE:
            found.setdefault(str(item), _neighbor_reduction_of(item))
            continue
        stack.extend(item.operands())
    return list(found.values())
