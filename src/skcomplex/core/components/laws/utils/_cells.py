"""Define the exterior calculus of the states on the cells of a cell complex."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression

from ._operators import _OperatorFunction
from ._reductions import _calls_in

CELL_OPERATIONS = ('exterior_derivative', 'codifferential', 'hodge_star')
EXTERIOR_DERIVATIVE = _OperatorFunction('exterior_derivative', nargs=1, latex_name=r'\mathrm{d}')
CODIFFERENTIAL = _OperatorFunction('codifferential', nargs=1, latex_name=r'\delta')
HODGE_STAR = _OperatorFunction('hodge_star', nargs=1, latex_name=r'\star')


def exterior_derivative(expression: object) -> Expression:
    """Return the differences of a state of a cell complex, one degree up.

    The degree of a state is where its group lives, 0 on sites, 1 on edges, and 2 on faces, which a group holds as
    hyperedges. On a state of the sites it gives every edge the value at its head minus the value at its tail, and on a
    state of the edges it gives every face the sum around it, each edge signed by its direction, so applying it twice
    gives exactly 0. Inside a reduction over `CofaceNeighborhood` it is the signed sum around the one face, such as
    `exterior_derivative(sin(theta))` for the differences of phases along every edge. It gives the voltage across the
    wires of a circuit, the pressure drop along a pipe network, and the gradient of a field on a mesh.

    Read more in the user guide, under Laws.

    Args:
        expression:
            An expression of the states and parameters of a group on the cells of a cell complex.

    Returns:
        derivative:
            The call `exterior_derivative(expression)`.

    See Also:
        codifferential: Goes one degree down.
        hodge_star: Scales a state by the sizes of its cells.

    Examples:
        >>> from sage.all__sagemath_symbolics import sin, var
        >>> from skcomplex.core.components.laws.utils import exterior_derivative
        >>> exterior_derivative(sin(var('theta')))
        exterior_derivative(sin(theta))
    """
    return EXTERIOR_DERIVATIVE(SR(expression))


def codifferential(expression: object) -> Expression:
    """Return the net flow of a state of a cell complex, one degree down.

    It is `(-1)^k star^-1 d^T star` for a state of degree `k`, where `d^T` turns the differences of
    `exterior_derivative` around. On a lattice with unit spacings it gives every site the outflow minus the inflow of a
    state of the edges, such as `codifferential(flow)`, and every edge the circulation of a state of the faces. Applied
    twice it gives 0 up to rounding, and applied to the differences of a state of the sites it gives the graph
    Laplacian. It gives the net current into the nodes of a circuit, the net trade of every country, and the divergence
    of a flow on a mesh.

    Read more in the user guide, under Laws.

    Args:
        expression:
            An expression of the states and parameters of a group on the edges or faces of a cell complex.

    Returns:
        codifferential:
            The call `codifferential(expression)`.

    See Also:
        exterior_derivative: Goes one degree up.
        hodge_star: Scales a state by the sizes of its cells.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import codifferential
        >>> codifferential(var('flow'))
        codifferential(flow)
    """
    return CODIFFERENTIAL(SR(expression))


def hodge_star(expression: object) -> Expression:
    """Return a state of a cell complex scaled cell by cell, the Hodge star.

    It multiplies the value of every cell by a factor of that cell, such as `hodge_star(flow)` for the currents that the
    voltages of the edges drive. On a lattice the factor is the size of the dual cell over the size of the cell, from
    the spacings. On a network it is the parameter `weight` of the elements of the group on those cells, as the system
    holds it when it compiles, or 1 for a group without that parameter. It gives the conductance of the wires of a
    circuit, the capacity of the pipes of a network, and the areas of the cells of a mesh.

    Read more in the user guide, under Laws.

    Args:
        expression:
            An expression of the states and parameters of a group on the cells of a cell complex.

    Returns:
        star:
            The call `hodge_star(expression)`.

    See Also:
        exterior_derivative: Takes the differences of a state, one degree up.
        codifferential: Takes the net flow of a state, one degree down.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import hodge_star
        >>> hodge_star(var('flow'))
        hodge_star(flow)
    """
    return HODGE_STAR(SR(expression))


def cell_operations_in(expression: object) -> list[Expression]:
    """Return the calls of the operations of a cell complex in an expression.

    The compiler reads them to build a matrix for every call of `exterior_derivative`, `codifferential`, and
    `hodge_star`, each listed once, such as the outer call in `1 + codifferential(exterior_derivative(flow))`. A call
    inside another call is left for the outer one. It finds the operations of circuits, pipe networks, and meshes.

    Read more in the user guide, under Extend the library.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        calls:
            The calls of `exterior_derivative`, `codifferential`, and `hodge_star`.

    See Also:
        exterior_derivative: Builds one of the calls it finds.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import codifferential, exterior_derivative
        >>> from skcomplex.core.components.laws.utils._cells import cell_operations_in
        >>> cell_operations_in(1 + codifferential(exterior_derivative(var('flow'))))
        [codifferential(exterior_derivative(flow))]
    """
    return _calls_in(expression, set(CELL_OPERATIONS))
