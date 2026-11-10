"""Define the contraction of a coupling tensor with expressions of the members."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression

from ....errors import ValidationError
from ._operators import _OperatorFunction
from ._reductions import _calls_in

MIN_FACTORS = 1
CONTRACTION = _OperatorFunction('contraction')


def contraction(tensor: object, *expressions: object) -> Expression:
    """Return the contraction of a tensor with one expression per later index.

    It is `sum_jk T_ijk a_j b_k`, with one expression per index after the first. The tensor is a parameter of an
    interaction declared `ParameterSpec(shape=(role,) * k)`, whose first index runs over the reading members and every
    other index over the members of a group. Inside the contraction the symbols of the expression of an index stand for
    the member of that index, so `contraction(T, x, x)` reads `x_j` and `x_k`.

    Args:
        tensor:
            The symbol of the shaped parameter of order `k`.

        *expressions:
            The `k - 1` expressions of one member, one per index after the first.

    Returns:
        contraction:
            The unevaluated contraction, one value per reading member, which every backend lowers.

    Raises:
        ValidationError:
            The tensor is not a symbol, or no expression is given.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import contraction
        >>> from skcomplex.core.components.laws.utils._contractions import contractions_in
        >>> T, x = var('T x')
        >>> contraction(T, x, x)
        contraction(T, x, x)
        >>> [call.operands()[0] for call in contractions_in(x * contraction(T, x, x))]
        [T]
    """
    if not isinstance(tensor, Expression) or not tensor.is_symbol():
        error_msg = f'The tensor of a contraction must be the symbol of a shaped parameter, not {tensor!r}.'
        raise ValidationError(error_msg)
    if len(expressions) < MIN_FACTORS:
        error_msg = f'The contraction of {tensor} needs one expression per index after the first, but none is given.'
        raise ValidationError(error_msg)
    return CONTRACTION(tensor, *(SR(item) for item in expressions))


def contractions_in(expression: object) -> list[Expression]:
    """Return the outermost contraction calls of an expression.

    Their operands are the tensor and the expressions.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        contractions:
            The contraction calls, each listed once.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import contraction
        >>> from skcomplex.core.components.laws.utils._contractions import contractions_in
        >>> T, x = var('T x')
        >>> len(contractions_in(1 + contraction(T, x)))
        1
    """
    return _calls_in(expression, {'contraction'})
