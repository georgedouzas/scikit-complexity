"""Define the tensor operations of fields.

They are compositions of the field operators with Sage matrices.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from math import prod

from sage.all__sagemath_symbolics import SR, Expression, matrix, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.structure.element import Matrix
from sage.symbolic.operators import add_vararg, mul_vararg

from ....errors import TensorShapeError
from ._operators import (
    CONVOLVE,
    INVERSE_HELMHOLTZ,
    SHIFTED,
    advection,
    divergence,
    gradient,
    inverse_laplacian,
    laplacian,
    partial,
    upwind,
)

LINEAR_OPERATORS = (laplacian, partial, upwind, CONVOLVE, SHIFTED)
UNBOUNDED_INVERSION_OPERANDS = 2


def _check_matrix(utility: str, value: object) -> Matrix:
    """Return a square matrix, or raise an error that names the utility and the value."""
    if not isinstance(value, Matrix) or not value.is_square():
        error_msg = f'{utility} needs a square matrix, but got {value}, which has another shape.'
        raise TensorShapeError(error_msg)
    return value


def _check_vector(utility: str, value: object) -> FreeModuleElement:
    """Return a vector, or a sequence as a vector, or raise an error.

    The error names the utility and the value.
    """
    if isinstance(value, list | tuple):
        return vector(SR, [SR(item) for item in value])
    if not isinstance(value, FreeModuleElement):
        error_msg = f'{utility} needs a vector, but got {value}, which has another shape.'
        raise TensorShapeError(error_msg)
    return value


def _derivative(expression: object, axis: int) -> Expression:
    """Return the derivative of an entry of a matrix along an axis, 0 for a number."""
    item = SR(expression)
    return SR(0) if item.is_numeric() else partial(item, SR(axis))


def _advected(expression: Expression, velocity: FreeModuleElement) -> Expression:
    """Return the advection of an entry of a matrix.

    It is 0 for a number, whose derivatives are 0.
    """
    return SR(0) if SR(expression).is_numeric() else advection(expression, velocity)


def _distribute_call(call: Expression, operands: list[Expression]) -> Expression:
    """Return a call of a linear operator as a sum over the terms of its first operand.

    The numeric factors stay outside.
    """
    operator = call.operator()
    expanded = operands[0].expand()
    total = SR(0)
    for term in expanded.operands() if expanded.operator() is add_vararg else [expanded]:
        factors = term.operands() if term.operator() is mul_vararg else [term]
        coefficient = prod((factor for factor in factors if factor.is_numeric()), start=SR(1))
        total += coefficient * operator(term / coefficient, *operands[1:])
    return total


def _is_linear_call(expression: Expression) -> bool:
    """Check whether an expression calls an operator linear in its first operand."""
    operator = expression.operator()
    if operator == INVERSE_HELMHOLTZ:
        return len(expression.operands()) == UNBOUNDED_INVERSION_OPERANDS
    return any(operator == item for item in LINEAR_OPERATORS)


def distribute_operators(expression: object) -> Expression:
    """Return an expression with the linear field operators distributed.

    They are distributed over sums and out of numeric factors.

    Args:
        expression:
            An expression of the states and parameters of a field.

    Returns:
        expression:
            The same field, in which every operator reads a product of symbols, such as `laplacian(-a)` as
            `-laplacian(a)`.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import distribute_operators, laplacian
        >>> a = var('a')
        >>> distribute_operators(laplacian(-a) + laplacian(a))
        0
    """
    item = SR(expression)
    operator = item.operator()
    if operator is None:
        return item
    operands = [distribute_operators(operand) for operand in item.operands()]
    return _distribute_call(item, operands) if _is_linear_call(item) else operator(*operands)


def vector_gradient(velocity: Sequence[object] | FreeModuleElement) -> Matrix:
    """Return the gradient of a vector.

    It is the matrix of the derivatives `partial(v_i, j)`.

    Args:
        velocity:
            One expression per dimension.

    Returns:
        gradient:
            The matrix of the derivative of every component along every dimension.

    Raises:
        TensorShapeError:
            The value is not a vector.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import vector_gradient
        >>> vector_gradient([var('u'), var('v')])
        [partial(u, 0) partial(u, 1)]
        [partial(v, 0) partial(v, 1)]
    """
    components = _check_vector('vector_gradient', velocity)
    size = len(components)
    return matrix(SR, [[_derivative(item, axis) for axis in range(size)] for item in components])


def tensor_divergence(tensor: Matrix) -> FreeModuleElement:
    """Return the divergence of a rank-2 tensor.

    It is the vector of the sums of `partial(T_ij, j)` over `j`.

    Args:
        tensor:
            A square matrix of expressions.

    Returns:
        divergence:
            One expression per row of the tensor.

    Raises:
        TensorShapeError:
            The value is not a square matrix.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, identity_matrix, var
        >>> from skcomplex.core.components.laws.utils import tensor_divergence
        >>> tensor_divergence(var('p') * identity_matrix(SR, 2))
        (partial(p, 0), partial(p, 1))
    """
    rows = _check_matrix('tensor_divergence', tensor).rows()
    return vector(SR, [sum((_derivative(item, axis) for axis, item in enumerate(row)), SR(0)) for row in rows])


def symmetric_part(tensor: Matrix) -> Matrix:
    """Return the symmetric part of a square matrix, `(M + M^T) / 2`.

    Args:
        tensor:
            A square matrix of expressions.

    Returns:
        part:
            The symmetric matrix.

    Raises:
        TensorShapeError:
            The value is not a square matrix.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, matrix, var
        >>> from skcomplex.core.components.laws.utils import symmetric_part
        >>> symmetric_part(matrix(SR, [[0, 2 * var('a')], [0, 0]]))
        [0 a]
        [a 0]
    """
    checked = _check_matrix('symmetric_part', tensor)
    return (checked + checked.transpose()) / 2


def antisymmetric_part(tensor: Matrix) -> Matrix:
    """Return the antisymmetric part of a square matrix, `(M - M^T) / 2`.

    Args:
        tensor:
            A square matrix of expressions.

    Returns:
        part:
            The antisymmetric matrix.

    Raises:
        TensorShapeError:
            The value is not a square matrix.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, matrix, var
        >>> from skcomplex.core.components.laws.utils import antisymmetric_part
        >>> antisymmetric_part(matrix(SR, [[0, 2 * var('a')], [0, 0]]))
        [ 0  a]
        [-a  0]
    """
    checked = _check_matrix('antisymmetric_part', tensor)
    return (checked - checked.transpose()) / 2


def leray_projection(field: Sequence[object] | FreeModuleElement) -> FreeModuleElement:
    """Return the divergence-free part of a vector of field expressions.

    It is `v - grad(inverse_laplacian(div(v)))`.

    Args:
        field:
            One expression per dimension.

    Returns:
        projection:
            The divergence-free vector.

    Raises:
        TensorShapeError:
            The value is not a vector.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import leray_projection
        >>> len(leray_projection([var('u'), var('v'), var('w')]))
        3
    """
    components = _check_vector('leray_projection', field)
    potential = inverse_laplacian(divergence(components))
    return components - gradient(potential, len(components))
