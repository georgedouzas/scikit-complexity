"""Define the operators of quantum subsystems, as Sage symbolic functions the compiler lowers to matrices.

Sage symbolic products commute, but operators do not, so each local operator, each ordered product of operators, and
each adjoint is a Sage function made by `function`, kept unevaluated in expressions. A local operator names its
subsystem and the operator of the subsystem's `HilbertSpec`. Coefficients are ordinary Sage expressions, so an operator
is a sum of numbers or parameter expressions times local operators, products, and adjoints, `J * operator_product(a,
b)` rather than `J * a * b`.

Examples:
    >>> from sage.all__sagemath_symbolics import SR, var
    >>> from skcomplex.core.laws.utils import dagger, local_operator, operator_product
    >>> a = local_operator(SR.symbol('a'), SR.symbol('x'))
    >>> b = local_operator(SR.symbol('b'), SR.symbol('minus'))
    >>> var('J') * operator_product(a, b) + dagger(b)
    J*operator_product(local_operator(a, x), local_operator(b, minus)) + dagger(local_operator(b, minus))
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression, function

local_operator = function('local_operator', nargs=2)
operator_product = function('operator_product')
dagger = function('dagger', nargs=1)


def has_operator(expression: object) -> bool:
    """Check whether an expression holds a local operator, which makes it an operator rather than a scalar.

    Args:
        expression:
            A Sage expression, or any other value.

    Returns:
        has_operator:
            Whether a local operator appears in the expression.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, var
        >>> from skcomplex.core.laws.utils import has_operator, local_operator
        >>> has_operator(2 * local_operator(SR.symbol('q'), SR.symbol('z'))), has_operator(var('J'))
        (True, False)
    """
    return isinstance(expression, Expression) and expression.has(local_operator(SR.wild(0), SR.wild(1)))
