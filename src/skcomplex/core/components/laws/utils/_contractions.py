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
    """Return the sum of a coupling tensor times values of the members of a group.

    It gives every member `i` the sum `Σ_jk T_ijk a_j b_k`, with one expression per index after the first, such as
    `contraction(T, x, x)` for the sum of `T_ijk x_j x_k` over all pairs of members `j` and `k`. The tensor is a
    parameter of an interaction, an array with one axis per index, whose first index runs over the members that read the
    sum. Inside the sum the values of each expression are those of the member of its index. It couples triples of
    neurons in a network with higher order links, of species in a community, and of spins in a model with three body
    terms.

    Read more in the user guide, under Laws.

    Args:
        tensor:
            The symbol of the array parameter, with `k` axes.

        *expressions:
            The `k - 1` expressions of one member, one per index after the first.

    Returns:
        contraction:
            The unevaluated contraction, one value per reading member, which every backend lowers.

    Raises:
        ValidationError:
            The tensor is not a symbol, or no expression is given.

    See Also:
        bilinear: Sums a matrix times the values of two members.
        neighbors: Reduces the values of the neighbours of every member.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import contraction
        >>> T, x = var('T x')
        >>> contraction(T, x, x)
        contraction(T, x, x)
    """
    if not isinstance(tensor, Expression) or not tensor.is_symbol():
        error_msg = f'The tensor of a contraction must be the symbol of an array parameter, but it is {tensor!r}.'
        raise ValidationError(error_msg)
    if len(expressions) < MIN_FACTORS:
        error_msg = f'The contraction of {tensor} needs one expression per index after the first, but none is given.'
        raise ValidationError(error_msg)
    return CONTRACTION(tensor, *(SR(item) for item in expressions))


def contractions_in(expression: object) -> list[Expression]:
    """Return the calls of `contraction` in an expression, outermost first.

    The compiler reads them to build the sum of every call, whose operands are the tensor and then the expressions, such
    as one call in `1 + contraction(T, x)`. A call inside another call is left for the outer one. It finds the couplings
    of triples of neurons, species, and spins.

    Read more in the user guide, under Extend the library.

    Args:
        expression:
            A symbolic expression, a number, or a vector of expressions.

    Returns:
        contractions:
            The contraction calls, each listed once.

    See Also:
        contraction: Builds the calls it finds.

    Examples:
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import contraction
        >>> from skcomplex.core.components.laws.utils._contractions import contractions_in
        >>> T, x = var('T x')
        >>> len(contractions_in(1 + contraction(T, x)))
        1
    """
    return _calls_in(expression, {'contraction'})
