"""Define the exterior calculus of cell complexes.

It is written as Sage symbolic functions the compiler lowers.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression, function

from ._reductions import _calls_in

EXTERIOR_DERIVATIVE = function('exterior_derivative', nargs=1, latex_name=r'\mathrm{d}')
CODIFFERENTIAL = function('codifferential', nargs=1, latex_name=r'\delta')
HODGE_STAR = function('hodge_star', nargs=1, latex_name=r'\star')
CELL_OPERATIONS = ('exterior_derivative', 'codifferential', 'hodge_star')


def exterior_derivative(expression: object) -> Expression:
    """Return the exterior derivative of a state on the cells of a cell complex.

    It lives on the next degree. The degree of the expression is the part its group lives on, 0 for sites, 1 for edges,
    and 2 for hyperedges. On a site state it is the difference of the values at the ends of every edge, on an edge state
    the oriented sum around every face. Applied twice it is exactly 0. Inside a reduction over `CofaceNeighborhood` it
    is the oriented sum around the one coface.

    Args:
        expression:
            An expression of the states and parameters of a group on the cells of a cell complex.

    Returns:
        derivative:
            The call `exterior_derivative(expression)`.

    Examples:
        >>> from sage.all__sagemath_symbolics import sin, var
        >>> from skcomplex.core.components.laws.utils import exterior_derivative
        >>> exterior_derivative(sin(var('theta')))
        exterior_derivative(sin(theta))
    """
    return EXTERIOR_DERIVATIVE(SR(expression))


def codifferential(expression: object) -> Expression:
    """Return the codifferential `(-1)^k star^-1 d^T star` of a state of degree `k`.

    It lives on the degree below. Here `d^T` is the transpose of the incidence of the degrees `k - 1` and `k`, so on a
    lattice with unit spacings the codifferential of an edge state is its divergence, the outflow minus the inflow at
    every site, and the codifferential of a face state is its curl on every edge. Applied twice it is 0 up to roundoff.
    With this sign the codifferential of the exterior derivative of a site state is the graph Laplacian, and that of an
    edge state the curl of its curl.

    Args:
        expression:
            An expression of the states and parameters of a group on the edges or hyperedges of a cell complex.

    Returns:
        codifferential:
            The call `codifferential(expression)`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import codifferential
        >>> codifferential(var('flow'))
        codifferential(flow)
    """
    return CODIFFERENTIAL(SR(expression))


def hodge_star(expression: object) -> Expression:
    """Return the Hodge star of a state on the cells of a cell complex.

    It is a diagonal factor of every cell. On a lattice complex the factor is the ratio of dual to primal cell volumes
    from the spacings. On a network complex it is the parameter `weight` of the elements of the group on the part of the
    cell, as the group holds it when the system compiles, and 1 for a part whose group declares no such parameter.

    Args:
        expression:
            An expression of the states and parameters of a group on the cells of a cell complex.

    Returns:
        star:
            The call `hodge_star(expression)`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import hodge_star
        >>> hodge_star(var('flow'))
        hodge_star(flow)
    """
    return HODGE_STAR(SR(expression))


def cell_operations_in(expression: object) -> list[Expression]:
    """Return the outermost calls of the exterior calculus in an expression.

    Each is listed once.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        calls:
            The calls of `exterior_derivative`, `codifferential`, and `hodge_star`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import cell_operations_in, codifferential, exterior_derivative
        >>> cell_operations_in(1 + codifferential(exterior_derivative(var('flow'))))
        [codifferential(exterior_derivative(flow))]
    """
    return _calls_in(expression, set(CELL_OPERATIONS))
