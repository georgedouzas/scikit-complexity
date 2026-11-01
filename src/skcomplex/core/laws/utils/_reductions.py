"""Define the reductions and reads across the members of a group, as Sage symbolic functions the compiler lowers.

Sage has no reduction over the members of a group, so each reduction is a Sage function made by `function`, kept
unevaluated in expressions. Arguments that are not symbolic, such as a constant weight matrix or a neighbourhood, ride
in the expression as a symbol named after a digest of their value.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import hashlib
import numbers
from types import SimpleNamespace

import numpy as np
import scipy.sparse
from sage.all__sagemath_symbolics import SR, Expression, arctan2, cos, function, sin, var, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.symbolic.function import Function

from ....errors import ValidationError
from ....specs import HyperedgeReductionSpec, LayerSpec, NeighborReductionSpec, ReductionSpec
from ....specs._draw_spec import draws_in
from ...spaces import AdjacencyNeighborhood, BaseNeighborhood, LatticeNeighborhood, MemoryNeighborhood
from ...spaces._lattice import BOUNDARIES
from ._conditions import both
from ._pointwise import _largest, _smallest

KINDS = {'total': 'sum', 'mean': 'mean', 'maximum': 'max', 'minimum': 'min', 'variance': 'var', 'count': 'count'}
NEIGHBOR_REDUCTIONS = ('sum', 'mean', 'count', 'max', 'min', 'normalized-mean', 'argmax')
MIN_HYPEREDGE_SIZE = 2
COMPONENT_REDUCTIONS = ('label', 'size', 'largest', 'giant')
WEIGHTS: dict[str, object] = {}
NEIGHBORHOODS: dict[str, object] = {}


def _is_placeholder(item: Expression, value: int) -> bool:
    """Check whether an operand is the number that stands for an absent condition, weights, group, or neighbourhood."""
    item = SR(item)
    return bool(item.is_numeric() and item == value)


def _print_reduction(
    self: Function,
    argument: Expression,
    condition: Expression,
    weights: Expression,
    over: Expression,
) -> str:
    """Return a reduction call with its keywords, leaving out the absent condition, weights, and group."""
    name = self.name()
    parts = [] if name == 'count' else [str(argument)]
    if not _is_placeholder(over, 0):
        parts.append(f'over={over}')
    if not _is_placeholder(condition, 1):
        parts.append(f'where={condition}')
    if not _is_placeholder(weights, 0):
        parts.append(f'weights={weights}')
    return f'{name}({", ".join(parts)})'


def _print_neighbors(self: object, argument: Expression, weight: Expression, over: Expression, kind: Expression) -> str:
    """Return a neighbourhood reduction call with its keywords, leaving out the own neighbourhood and unit weight."""
    del self
    parts = [str(argument)]
    if not _is_placeholder(over, 0):
        parts.append(f'over={NEIGHBORHOODS.get(str(over), over)!r}')
    parts.append(f'reduce={NEIGHBOR_REDUCTIONS[int(SR(kind))]!r}')
    if not _is_placeholder(weight, 1):
        parts.append(f'weight={weight}')
    return f'neighbors({", ".join(parts)})'


def _print_hyperedges(
    self: object,
    argument: Expression,
    weight: Expression,
    size: Expression,
    kind: Expression,
) -> str:
    """Return a hyperedge reduction call with its keywords, leaving out every size and unit weight."""
    del self
    parts = [str(argument), f'reduce={NEIGHBOR_REDUCTIONS[int(SR(kind))]!r}']
    if not _is_placeholder(size, 0):
        parts.append(f'size={int(SR(size))}')
    if not _is_placeholder(weight, 1):
        parts.append(f'weight={weight}')
    return f'hyperedges({", ".join(parts)})'


def _print_components(self: object, active: Expression, over: Expression, kind: Expression) -> str:
    """Return a components reduction call with its keywords, leaving out the own neighbourhood."""
    del self
    parts = [str(active)]
    if not _is_placeholder(over, 0):
        parts.append(f'over={NEIGHBORHOODS.get(str(over), over)!r}')
    parts.append(f'reduce={COMPONENT_REDUCTIONS[int(SR(kind))]!r}')
    return f'components({", ".join(parts)})'


def _eval_shift(
    self: object,
    expression: Expression,
    offset: Expression,
    boundary: Expression,
    fill: Expression,
) -> Expression | None:
    """Return the expression itself for an offset of 0 or an expression without symbols or draws, or `None`."""
    del self, boundary, fill
    if _is_placeholder(offset, 0) or not (SR(expression).variables() or draws_in(expression)):
        return expression
    return None


REDUCTIONS = {name: function(name, nargs=4, print_func=_print_reduction) for name in KINDS}
NEIGHBORS = function('neighbors', nargs=4, print_func=_print_neighbors)
NEIGHBOR = function('neighbor', nargs=2)
HYPEREDGES = function('hyperedges', nargs=4, print_func=_print_hyperedges)
COMPONENTS = function('components', nargs=3, print_func=_print_components)
SHIFT_MEMBER = function('shift_member', nargs=4, eval_func=_eval_shift)


def _over_symbol(over: object) -> Expression:
    """Return a symbol that names the group of a reduction, from a namespace or a symbol, or 0 for none."""
    if over is None:
        return SR(0)
    if isinstance(over, SimpleNamespace):
        for name, value in vars(over).items():
            if name == 't':
                continue
            if isinstance(value, FreeModuleElement):
                return SR(next(iter(value)))
            if isinstance(value, Expression) and value.is_symbol():
                return value
    if isinstance(over, Expression) and over.is_symbol():
        return over
    error_msg = (
        'A reduction takes the group to reduce over as a namespace such as `namespace.members`, or as one of its '
        f'symbols, not {over!r}.'
    )
    raise ValidationError(error_msg)


def _weights_symbol(weights: object) -> Expression:
    """Return the symbol that stands for a constant weight matrix, or 0 for none."""
    if weights is None:
        return SR(0)
    if scipy.sparse.issparse(weights):
        matrix = scipy.sparse.csr_array(weights, dtype=float)
        parts = [np.asarray(matrix.shape), matrix.indptr, matrix.indices, matrix.data]
    else:
        matrix = np.asarray(weights, dtype=float)
        parts = [np.asarray(matrix.shape), np.ascontiguousarray(matrix)]
    if matrix.ndim not in {1, 2}:
        error_msg = (
            'The weights of a reduction are a matrix with one row per reading member, or one row that every reading '
            f'member shares, not of shape {matrix.shape}.'
        )
        raise ValidationError(error_msg)
    name = f'_weights_{hashlib.sha256(b"".join(np.asarray(part).tobytes() for part in parts)).hexdigest()[:16]}'
    WEIGHTS[name] = matrix
    return var(name)


def _keyed(over: object, where: object, by: str | None) -> object:
    """Return the condition of a reduction, with the members that share the reading member's value of a state."""
    if by is None:
        return where
    reduced = getattr(getattr(over, 'other', None), by, None) if isinstance(over, SimpleNamespace) else None
    if not isinstance(by, str) or reduced is None:
        error_msg = (
            'A reduction keyed by a state takes the name of a state of its group as `by` and the namespace of the '
            f'group as `over`, such as `over=namespace.members, by=\'cell\'`, not by={by!r} over={over!r}.'
        )
        raise ValidationError(error_msg)
    shared = SR(reduced) == SR(getattr(over, by))
    return shared if where is None else both(where, shared)


def _reduce(
    name: str,
    expression: object,
    over: object,
    where: object,
    weights: object,
    *,
    by: str | None = None,
) -> Expression:
    """Return the unevaluated call of a reduction, with placeholders for the absent condition, weights, and group."""
    where = _keyed(over, where, by)
    condition = SR(1) if where is None else SR(where)
    return REDUCTIONS[name](SR(expression), condition, _weights_symbol(weights), _over_symbol(over))


def total(
    expression: object,
    over: object = None,
    where: object = None,
    weights: object = None,
    by: str | None = None,
) -> Expression:
    """Return the sum of an expression over the members of a group, `sum_j a_j`, or `sum_j W_ij a_j` with weights.

    Inside the reduction, the symbols of a member stand for the reduced member, and outside it for the reading member.

    Args:
        expression:
            The expression of one member, such as `namespace.x` or `cos(namespace.members.theta)`.

        over:
            The namespace of the group to reduce over, such as `namespace.members`, or `None` for the group the
            expression reads.

        where:
            A condition on the reduced member that selects the members, or `None` for all members.

        weights:
            A constant matrix, a NumPy array or a SciPy sparse matrix, with one row per reading member and one column
            per reduced member, a vector with one weight per reduced member that every reading member shares, or
            `None`.

        by:
            The name of a state of the group, so every member reduces over the members that share its value of the
            state, or `None` for all members. It needs the namespace of the group as `over`.

    Returns:
        reduction:
            The unevaluated reduction, which every backend lowers.

    Raises:
        ValidationError:
            The group is not a namespace or a symbol, the weights are not a matrix, or a key has no group namespace.

    Examples:
        >>> from sage.all__sagemath_symbolics import tanh, var
        >>> from skcomplex.core.laws.utils import total
        >>> x = var('x')
        >>> -x + tanh(total(x))
        -x + tanh(total(x))
        >>> total(x, where=x > 0.5)
        total(x, where=x > 0.5)
    """
    return _reduce('total', expression, over, where, weights, by=by)


def mean(
    expression: object,
    over: object = None,
    where: object = None,
    weights: object = None,
    by: str | None = None,
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

        by:
            The name of a state of the group, so every member reduces over the members that share its value of the
            state, or `None` for all members. It needs the namespace of the group as `over`.

    Returns:
        reduction:
            The unevaluated reduction, 0 when no member is selected.

    Raises:
        ValidationError:
            The group is not a namespace or a symbol, the weights are not a matrix, or a key has no group namespace.

    Examples:
        >>> from sage.all__sagemath_symbolics import cos, var
        >>> from skcomplex.core.laws.utils import mean
        >>> mean(cos(var('theta')))
        mean(cos(theta))
    """
    return _reduce('mean', expression, over, where, weights, by=by)


def maximum(*expressions: object, over: object = None, where: object = None) -> Expression:
    """Return the largest value of an expression over the selected members of a group, or the largest of expressions.

    With one expression it reduces over the members of a group, as `max` over an array. With several it compares
    them member by member, as `max` over its arguments, with no group or condition.

    Args:
        *expressions:
            The expression of one member, or several expressions to compare member by member.

        over:
            The namespace of the group to reduce over, or `None` for the group the expression reads.

        where:
            A condition on the reduced member that selects the members, or `None` for all members.

    Returns:
        value:
            The unevaluated reduction, 0 when no member is selected, or the largest of the expressions.

    Raises:
        ValidationError:
            No expression is given, several expressions come with a group or a condition, or the group is not a
            namespace or a symbol.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import maximum
        >>> x, y = var('x y')
        >>> maximum(x)
        maximum(x)
        >>> maximum(x, y)
        where(y > x, y, x)
        >>> maximum(x, y).subs(x=1, y=2)
        2
    """
    if len(expressions) == 1:
        return _reduce('maximum', expressions[0], over, where, None)
    if not expressions or over is not None or where is not None:
        error_msg = (
            'maximum takes one expression to reduce over a group, or several expressions to compare member by member '
            f'without a group or a condition, but got {len(expressions)} expressions.'
        )
        raise ValidationError(error_msg)
    return _largest(expressions)


def minimum(*expressions: object, over: object = None, where: object = None) -> Expression:
    """Return the smallest value of an expression over the selected members of a group, or the smallest of expressions.

    With one expression it reduces over the members of a group, as `min` over an array. With several it compares
    them member by member, as `min` over its arguments, with no group or condition.

    Args:
        *expressions:
            The expression of one member, or several expressions to compare member by member.

        over:
            The namespace of the group to reduce over, or `None` for the group the expression reads.

        where:
            A condition on the reduced member that selects the members, or `None` for all members.

    Returns:
        value:
            The unevaluated reduction, 0 when no member is selected, or the smallest of the expressions.

    Raises:
        ValidationError:
            No expression is given, several expressions come with a group or a condition, or the group is not a
            namespace or a symbol.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import minimum
        >>> x, y = var('x y')
        >>> minimum(x)
        minimum(x)
        >>> minimum(x, y)
        where(y < x, y, x)
        >>> minimum(x, y).subs(x=1, y=2)
        1
    """
    if len(expressions) == 1:
        return _reduce('minimum', expressions[0], over, where, None)
    if not expressions or over is not None or where is not None:
        error_msg = (
            'minimum takes one expression to reduce over a group, or several expressions to compare member by member '
            f'without a group or a condition, but got {len(expressions)} expressions.'
        )
        raise ValidationError(error_msg)
    return _smallest(expressions)


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
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import variance
        >>> variance(var('x'))
        variance(x)
    """
    return _reduce('variance', expression, over, where, None)


def count(over: object = None, where: object = None, by: str | None = None) -> Expression:
    """Return the number of members of a group that a condition selects, or of all members.

    Args:
        over:
            The namespace of the group to count, or `None` for the group the condition reads, or else the group of the
            law.

        where:
            A condition on the counted member, or `None` for all members.

        by:
            The name of a state of the group, so every member reduces over the members that share its value of the
            state, or `None` for all members. It needs the namespace of the group as `over`.

    Returns:
        reduction:
            The unevaluated count.

    Raises:
        ValidationError:
            The group is not a namespace or a symbol, or a key has no group namespace.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import count
        >>> x = var('x')
        >>> count(where=x > 0.5)
        count(where=x > 0.5)
        >>> from types import SimpleNamespace
        >>> members = SimpleNamespace(cell=var('cell'), other=SimpleNamespace(cell=var('cell_other')))
        >>> count(over=members, by='cell')
        count(over=cell, where=cell_other == cell)
    """
    return _reduce('count', 1, over, where, None, by=by)


def circular_mean(
    angle: object,
    over: object = None,
    where: object = None,
    weights: object = None,
    by: str | None = None,
) -> Expression:
    """Return the circular mean of an angle over the members of a group, `arctan2(sum_j sin a_j, sum_j cos a_j)`.

    The mean is the direction of the sum of the unit vectors at the angles, 0 when no member is selected.

    Args:
        angle:
            The angle of one member, in radians.

        over:
            The namespace of the group to reduce over, or `None` for the group the angle reads.

        where:
            A condition on the reduced member that selects the members, or `None` for all members.

        weights:
            A constant matrix with one row per reading member and one column per reduced member, or `None`.

        by:
            The name of a state of the group, so every member reduces over the members that share its value of the
            state, or `None` for all members. It needs the namespace of the group as `over`.

    Returns:
        reduction:
            The symbolic circular mean, in `[-pi, pi]`.

    Raises:
        ValidationError:
            The group is not a namespace or a symbol, the weights are not a matrix, or a key has no group namespace.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import circular_mean
        >>> circular_mean(var('theta'))
        arctan2(total(sin(theta)), total(cos(theta)))
    """
    sines = total(sin(SR(angle)), over, where, weights, by)
    cosines = total(cos(SR(angle)), over, where, weights, by)
    return arctan2(sines, cosines)


def _neighborhood_symbol(over: object) -> Expression:
    """Return the symbol that stands for a neighbourhood, registered by its value, or 0 for the space's own."""
    if over is None:
        return SR(0)
    if isinstance(over, LayerSpec):
        over = AdjacencyNeighborhood(layer=over.label)
    if not isinstance(over, BaseNeighborhood):
        error_msg = f'A neighbourhood reduction reduces over None, a neighbourhood, or a LayerSpec, not {over!r}.'
        raise ValidationError(error_msg)
    name = f'_over_{hashlib.sha256(repr(over).encode()).hexdigest()[:16]}'
    NEIGHBORHOODS[name] = over
    return var(name)


def neighbors(
    expression: object,
    over: BaseNeighborhood | LayerSpec | None = None,
    reduce: str = 'sum',
    weight: object = None,
) -> Expression | FreeModuleElement:
    """Return the reduction of an expression over the neighbours of every member of a group.

    The expression is written for one neighbour: the namespace `namespace.other` holds the symbols of the neighbour,
    and `namespace` those of the member that reads the reduction, so `sin(namespace.other.theta - namespace.theta)` is
    the phase difference to one neighbour. The neighbourhood is the graph of a network space by default,
    `AdjacencyNeighborhood()`, a layer of the network, `AdjacencyNeighborhood(layer=label)` or the `LayerSpec` itself, a
    lattice neighbourhood, or in a continuous space the members within a distance, `RadiusNeighborhood(r)`, the nearest
    members, `NearestNeighborhood(k)`, or the members within a distance inside a cone around a heading,
    `ConeNeighborhood(r, angle)`. A reduction over no neighbours is 0.

    The reductions with the weight `w_j` of every neighbour `j` of the value `f_j` are `'sum'`, `sum_j w_j f_j`,
    `'mean'`, the sum divided by the number of neighbours, `'count'`, `sum_j w_j [f_j != 0]`, `'max'` and `'min'`, the
    largest and smallest `w_j f_j`, `'normalized-mean'`, `sum_j w_j f_j / sum_j w_j`, and `'argmax'`, the value `f_j` of
    the neighbour whose weight `w_j` is largest, the first of them in neighbour order. The circular mean of an angle,
    `'circular-mean'`, is `arctan2(sum_j w_j sin f_j, sum_j w_j cos f_j)`, 0 over no neighbours. On a lattice
    neighbourhood, `reduce='each'` gives the values of a lattice state at every offset, in neighbourhood order, without
    reducing them.

    Args:
        expression:
            The value of one neighbour, an expression of the symbols of `namespace.other` and `namespace`, or a vector
            of them.

        over:
            The neighbourhood, or `None` for the neighbours of the network or lattice the group lives on.

        reduce:
            `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, `'normalized-mean'`, `'argmax'`, or `'circular-mean'`, or
            `'each'` for the values at every offset of a lattice neighbourhood or a memory.

        weight:
            The weight of one neighbour, an expression of the same symbols, or `None` for 1. The `'argmax'` reduction
            picks the neighbour whose weight is largest.

    Returns:
        reduction:
            The unevaluated reduction, one per component of a vector expression, or with `reduce='each'` the values of
            a lattice state at the offsets of a lattice neighbourhood.

    Raises:
        ValidationError:
            The reduction or the neighbourhood is unknown.

    Examples:
        >>> from sage.all__sagemath_symbolics import sin, var
        >>> from skcomplex.core.laws.utils import neighbors
        >>> from skcomplex.core.spaces import RadiusNeighborhood, VonNeumannNeighborhood
        >>> neighbors(sin(var('theta_other')), over=RadiusNeighborhood(1.0), reduce='mean')
        neighbors(sin(theta_other), over=RadiusNeighborhood(radius=1.0), reduce='mean')
        >>> neighbors(var('strategy_other'), reduce='argmax', weight=var('payoff_other'))
        neighbors(strategy_other, reduce='argmax', weight=payoff_other)
        >>> values = neighbors(var('spin'), VonNeumannNeighborhood(), reduce='each')
        >>> len(values), values[0]
        (4, neighbor(spin, 0))
    """
    if reduce == 'each':
        if isinstance(over, LatticeNeighborhood):
            size = len(over.check().prepare().offsets_)
        elif isinstance(over, MemoryNeighborhood):
            size = len(over.check().prepare().elements_)
        else:
            error_msg = f"The reduction 'each' reads a lattice or a memory neighbourhood, not {over!r}."
            raise ValidationError(error_msg)
        return vector(SR, [NEIGHBOR(expression, index) for index in range(size)])
    if reduce == 'circular-mean':
        sines = neighbors(sin(SR(expression)), over, 'sum', weight)
        return arctan2(sines, neighbors(cos(SR(expression)), over, 'sum', weight))
    if reduce not in NEIGHBOR_REDUCTIONS:
        error_msg = (
            'Unknown neighbourhood reduction '
            f'{reduce!r}. Use one of {", ".join((*NEIGHBOR_REDUCTIONS, "circular-mean", "each"))}.'
        )
        raise ValidationError(error_msg)
    over_symbol = _neighborhood_symbol(over)
    code = SR(NEIGHBOR_REDUCTIONS.index(reduce))
    weighted = SR(1) if weight is None else SR(weight)
    if isinstance(expression, FreeModuleElement):
        return vector(SR, [NEIGHBORS(SR(item), weighted, over_symbol, code) for item in expression])
    return NEIGHBORS(SR(expression), weighted, over_symbol, code)


def hyperedges(
    expression: object,
    reduce: str = 'sum',
    size: int | None = None,
    weight: object = None,
) -> Expression | FreeModuleElement:
    """Return the reduction of an expression over the hyperedges around every member of a group on a network.

    The expression is written for one hyperedge around the member that reads the reduction: the namespaces
    `namespace.others` hold the symbols of the other members of the hyperedge in its order, `namespace.members` those
    of all its members, which `total`, `mean`, `count`, `maximum`, and `minimum` reduce with `over=namespace.members`,
    and `namespace` those of the reading member. So `sin(s.others[0].theta + s.others[1].theta - 2 * s.theta)` with
    `size=3` is the three-body phase coupling over the triangles around a member. The hyperedges are those of the
    network space of the group, and a reduction over no hyperedges is 0.

    The reductions are those of `neighbors` with the weight `w_e` of every hyperedge `e` of the value `f_e`: `'sum'`,
    `'mean'`, `'count'`, `'max'`, `'min'`, `'normalized-mean'`, `'argmax'`, and `'circular-mean'`. Without a size, the
    reduction runs over the hyperedges of every size, which the sum, the mean, and the count reduce, and the
    expression reads `namespace.members` rather than `namespace.others`.

    Args:
        expression:
            The value of one hyperedge, an expression of the symbols of `namespace.others`, `namespace.members`, and
            `namespace`, or a vector of them.

        reduce:
            `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, `'normalized-mean'`, `'argmax'`, or `'circular-mean'`.

        size:
            The number of members of the hyperedges, or `None` for every size.

        weight:
            The weight of one hyperedge, an expression of the same symbols, or `None` for 1.

    Returns:
        reduction:
            The unevaluated reduction, one per component of a vector expression.

    Raises:
        ValidationError:
            The reduction is unknown, or the size is not a whole number of at least two.

    Examples:
        >>> from sage.all__sagemath_symbolics import sin, var
        >>> from skcomplex.core.laws.utils import hyperedges
        >>> theta, first, second = var('theta first second')
        >>> hyperedges(sin(first + second - 2 * theta), size=3)
        hyperedges(sin(first + second - 2*theta), reduce='sum', size=3)
        >>> hyperedges(var('x'), reduce='mean')
        hyperedges(x, reduce='mean')
    """
    if size is not None and (
        isinstance(size, bool) or not isinstance(size, numbers.Integral) or size < MIN_HYPEREDGE_SIZE
    ):
        error_msg = f'A hyperedge reduction takes a size of 2 or more, or None for every size, not {size!r}.'
        raise ValidationError(error_msg)
    if reduce == 'circular-mean':
        sines = hyperedges(sin(SR(expression)), 'sum', size, weight)
        return arctan2(sines, hyperedges(cos(SR(expression)), 'sum', size, weight))
    if reduce not in NEIGHBOR_REDUCTIONS:
        error_msg = (
            f'Unknown hyperedge reduction {reduce!r}. Use one of {", ".join((*NEIGHBOR_REDUCTIONS, "circular-mean"))}.'
        )
        raise ValidationError(error_msg)
    code = SR(NEIGHBOR_REDUCTIONS.index(reduce))
    weighted = SR(1) if weight is None else SR(weight)
    sized = SR(0 if size is None else int(size))
    if isinstance(expression, FreeModuleElement):
        return vector(SR, [HYPEREDGES(SR(item), weighted, sized, code) for item in expression])
    return HYPEREDGES(SR(expression), weighted, sized, code)


def shift_member(expression: Expression, offset: int, *, boundary: str = 'periodic', fill: float = 0.0) -> Expression:
    """Return an expression of a group's members read at the member a number of places along the group order.

    The order of a group is the member index, so member `i` reads the expression at member `i + offset`. Past the
    ends, `'periodic'` wraps around to close a ring, `'reflecting'` mirrors the order so the end member reads itself
    for an offset of 1, and `'fixed'` reads `fill`. The compiler lowers the read to an index table of the group.

    Args:
        expression:
            An expression of the member symbols of one group.

        offset:
            The number of members along the group order, negative to read members behind.

        boundary:
            `'periodic'`, `'reflecting'`, or `'fixed'`.

        fill:
            The value read past the ends of a fixed boundary.

    Returns:
        shifted:
            The symbolic read at the offset member.

    Raises:
        ValidationError:
            The offset is not an integer or the boundary is unknown.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import shift_member
        >>> x = var('x')
        >>> shift_member(x, 1) - x
        -x + shift_member(x, 1, 0, 0.0)
        >>> shift_member(x, 0)
        x
    """
    if boundary not in BOUNDARIES or isinstance(offset, bool) or not isinstance(offset, numbers.Integral):
        error_msg = (
            f'shift_member takes a whole number of members and a boundary among {", ".join(BOUNDARIES)}, but '
            f'got the offset {offset!r} and the boundary {boundary!r}.'
        )
        raise ValidationError(error_msg)
    return SHIFT_MEMBER(expression, int(offset), BOUNDARIES.index(boundary), float(fill))


def _calls_in(expression: object, names: set[str]) -> list[Expression]:
    """Return the outermost calls of the named functions in an expression or a vector, each listed once."""
    if not isinstance(expression, Expression | FreeModuleElement | int | float):
        return []
    found: dict[str, Expression] = {}
    stack = [SR(item) for item in (list(expression) if isinstance(expression, FreeModuleElement) else [expression])]
    while stack:
        item = stack.pop()
        operator = item.operator()
        if operator is None:
            continue
        if getattr(operator, 'name', lambda: '')() in names:
            found.setdefault(str(item), item)
            continue
        stack.extend(item.operands())
    return list(found.values())


def reductions_in(expression: object) -> list[ReductionSpec]:
    """Return the outermost reduction calls of an expression, which may hold further reductions in their arguments.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        reductions:
            The reductions, each listed once.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import mean, reductions_in, total
        >>> x = var('x')
        >>> [item.kind for item in reductions_in(total(x * mean(x)))]
        ['sum']
    """
    reductions = []
    for call in _calls_in(expression, set(KINDS)):
        argument, condition, weights, over = call.operands()
        reductions.append(
            ReductionSpec(
                call=call,
                kind=KINDS[call.operator().name()],
                argument=argument,
                condition=None if _is_placeholder(condition, 1) else condition,
                weights=None if _is_placeholder(weights, 0) else WEIGHTS[str(weights)],
                over=None if _is_placeholder(over, 0) else over,
            ),
        )
    return reductions


def _component_reduction(call: Expression) -> NeighborReductionSpec:
    """Return the neighbourhood reduction of the links of a components call, of the kind `'component-'` and its name."""
    active, over, kind = call.operands()
    return NeighborReductionSpec(
        call=call,
        kind=f'component-{COMPONENT_REDUCTIONS[int(SR(kind))]}',
        argument=active,
        weight=None,
        over=None if _is_placeholder(over, 0) else NEIGHBORHOODS[str(over)],
    )


def neighborhoods_in(expression: object) -> list[NeighborReductionSpec]:
    """Return the outermost neighbourhood reductions of an expression, the components reductions included.

    A components reduction has the kind `'component-'` followed by its reduction, and the activity of one link as its
    argument.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        reductions:
            The neighbourhood reductions, each listed once.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import neighbors, neighborhoods_in
        >>> [item.kind for item in neighborhoods_in(neighbors(var('x'), reduce='mean'))]
        ['mean']
    """
    reductions = []
    for call in _calls_in(expression, {'neighbors', 'components'}):
        if call.operator().name() == 'components':
            reductions.append(_component_reduction(call))
            continue
        argument, weight, over, kind = call.operands()
        reductions.append(
            NeighborReductionSpec(
                call=call,
                kind=NEIGHBOR_REDUCTIONS[int(SR(kind))],
                argument=argument,
                weight=None if _is_placeholder(weight, 1) else weight,
                over=None if _is_placeholder(over, 0) else NEIGHBORHOODS[str(over)],
            ),
        )
    return reductions


def hyperedges_in(expression: object) -> list[HyperedgeReductionSpec]:
    """Return the outermost hyperedge reductions of an expression.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        reductions:
            The hyperedge reductions, each listed once.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.laws.utils import hyperedges, hyperedges_in
        >>> [(item.kind, item.size) for item in hyperedges_in(hyperedges(var('x'), reduce='mean', size=3))]
        [('mean', 3)]
    """
    reductions = []
    for call in _calls_in(expression, {'hyperedges'}):
        argument, weight, size, kind = call.operands()
        reductions.append(
            HyperedgeReductionSpec(
                call=call,
                kind=NEIGHBOR_REDUCTIONS[int(SR(kind))],
                argument=argument,
                weight=None if _is_placeholder(weight, 1) else weight,
                size=None if _is_placeholder(size, 0) else int(SR(size)),
            ),
        )
    return reductions
