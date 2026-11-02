"""Define the operators of subsystems, as Sage symbolic functions the compiler lowers to matrices.

Sage symbolic products commute, but operators do not, so each local operator, each ordered product of operators, and
each adjoint is a Sage function made by `function`, kept unevaluated in expressions. A local operator names its
subsystem and the operator of the subsystem's `HilbertSpec`. Coefficients are ordinary Sage expressions, so an operator
is a sum of numbers or parameter expressions times local operators, products, and adjoints, `J * operator_product(a,
b)` rather than `J * a * b`.

Examples:
    >>> from sage.all__sagemath_symbolics import SR, var
    >>> from skcomplex.core.laws.utils import dagger, local_operator, operator_product
    >>> a = local_operator(SR.symbol('a'), 'x')
    >>> b = local_operator(SR.symbol('b'), 'minus')
    >>> var('J') * operator_product(a, b) + dagger(b)
    J*operator_product(local_operator(a, x), local_operator(b, minus)) + dagger(local_operator(b, minus))
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression, function

LOCAL_OPERATOR = function('local_operator', nargs=2)
operator_product = function('operator_product')
dagger = function('dagger', nargs=1)


def local_operator(subsystem: object, name: object) -> Expression:
    """Return the operator of a name of the `HilbertSpec` of a subsystem, acting on that subsystem alone.

    Args:
        subsystem:
            The symbol of the state of the subsystem, such as `namespace.psi`.

        name:
            The name of the operator in the `HilbertSpec`, such as `'z'`, or its symbol.

    Returns:
        operator:
            The unevaluated local operator.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR
        >>> from skcomplex.core.laws.utils import local_operator
        >>> local_operator(SR.symbol('q'), 'z')
        local_operator(q, z)
    """
    return LOCAL_OPERATOR(subsystem, SR.symbol(name) if isinstance(name, str) else name)


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
        >>> has_operator(2 * local_operator(SR.symbol('q'), 'z')), has_operator(var('J'))
        (True, False)
    """
    return isinstance(expression, Expression) and expression.has(LOCAL_OPERATOR(SR.wild(0), SR.wild(1)))
