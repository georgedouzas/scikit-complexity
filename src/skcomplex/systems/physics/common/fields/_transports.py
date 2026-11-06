"""Define the transports of tensors and vectors in a flow, the co-rotational and upper-convected derivatives."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from sage.all__sagemath_symbolics import SR, Expression, identity_matrix, vector
from sage.modules.free_module_element import FreeModuleElement
from sage.structure.element import Matrix

from .....core.components.laws.utils import advection, antisymmetric_part, symmetric_part, vector_gradient
from .....core.errors import TensorShapeError


def _check_matrix(utility: str, value: object) -> Matrix:
    """Return a square matrix, or raise an error that names the utility and the value."""
    if not isinstance(value, Matrix) or not value.is_square():
        error_msg = f'{utility} needs a square matrix, but got {value}, which has another shape.'
        raise TensorShapeError(error_msg)
    return value


def _check_vector(utility: str, value: object) -> FreeModuleElement:
    """Return a vector, or a sequence as a vector, or raise an error that names the utility and the value."""
    if isinstance(value, list | tuple):
        return vector(SR, [SR(item) for item in value])
    if not isinstance(value, FreeModuleElement):
        error_msg = f'{utility} needs a vector, but got {value}, which has another shape.'
        raise TensorShapeError(error_msg)
    return value


def _advected(expression: Expression, velocity: FreeModuleElement) -> Expression:
    """Return the advection of an entry of a matrix, 0 for a number, whose derivatives are 0."""
    return SR(0) if SR(expression).is_numeric() else advection(expression, velocity)


def corotational(
    value: Matrix | FreeModuleElement,
    velocity: Sequence[object] | FreeModuleElement,
    xi: object = 0,
) -> Matrix | FreeModuleElement:
    """Return the transport of a symmetric tensor or a vector by a flow that rotates and aligns it.

    A symmetric tensor `T` gets the Beris-Edwards transport `u.grad(T) - S`, where
    `S = (xi E + W)(T + I/d) + (T + I/d)(xi E - W) - 2 xi (T + I/d) tr(T grad(u))`, with the strain rate `E`, the
    vorticity tensor `W`, the antisymmetric part of `vector_gradient(u)`, and the dimension `d`. A vector `p` gets
    `u.grad(p) - W p - xi E p`, so both turn with the flow.

    Args:
        value:
            A symmetric square matrix, or a vector.

        velocity:
            One velocity component per dimension.

        xi:
            The flow-alignment parameter.

    Returns:
        transport:
            The matrix or the vector to subtract from the rate of the value.

    Raises:
        TensorShapeError:
            The value is neither a square matrix nor a vector.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, matrix, var
        >>> from skcomplex.systems.physics.common.fields import corotational
        >>> a, b = var('a, b')
        >>> corotational(matrix(SR, [[a, b], [b, -a]]), [var('u'), 0]).dimensions()
        (2, 2)
    """
    flow = _check_vector('corotational', velocity)
    size = len(flow)
    gradients = vector_gradient(flow)
    strain, vorticity = symmetric_part(gradients), antisymmetric_part(gradients)
    if isinstance(value, FreeModuleElement):
        advected = vector(SR, [_advected(item, flow) for item in value])
        return advected - vorticity * value - SR(xi) * (strain * value)
    tensor = _check_matrix('corotational', value)
    advected_tensor = tensor.apply_map(lambda item: _advected(item, flow))
    shifted_tensor = tensor + identity_matrix(SR, size) / size
    aligned = (SR(xi) * strain + vorticity) * shifted_tensor + shifted_tensor * (SR(xi) * strain - vorticity)
    stretching = 2 * SR(xi) * (tensor * gradients).trace() * shifted_tensor
    return advected_tensor - aligned + stretching


def upper_convected(tensor: Matrix, velocity: Sequence[object] | FreeModuleElement) -> Matrix:
    """Return the upper-convected transport of a rank-2 tensor, `u.grad(T) - grad(u)^T T - T grad(u)`.

    Args:
        tensor:
            A square matrix of expressions.

        velocity:
            One velocity component per dimension.

    Returns:
        transport:
            The matrix to subtract from the rate of the tensor.

    Raises:
        TensorShapeError:
            The tensor is not a square matrix or the velocity is not a vector.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, identity_matrix, var
        >>> from skcomplex.systems.physics.common.fields import upper_convected
        >>> upper_convected(identity_matrix(SR, 2), [var('u'), var('v')])[0, 0]
        -2*partial(u, 0)
    """
    checked = _check_matrix('upper_convected', tensor)
    flow = _check_vector('upper_convected', velocity)
    gradients = vector_gradient(flow)
    advected = checked.apply_map(lambda item: _advected(item, flow))
    return advected - gradients.transpose() * checked - checked * gradients
