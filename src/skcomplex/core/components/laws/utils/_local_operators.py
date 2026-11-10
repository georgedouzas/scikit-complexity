"""Define the operators that act on one subsystem of a tensor product."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression

from ._operators import _OperatorFunction

LOCAL_OPERATOR = _OperatorFunction('local_operator', nargs=2)


operator_product = _OperatorFunction('operator_product')
dagger = _OperatorFunction('dagger', nargs=1)


def local_operator(subsystem: object, name: object) -> Expression:
    """Return the operator of a name of the `HilbertSpec` of a subsystem.

    It acts on that subsystem alone. Operators do not commute, so a product of operators is written with
    `operator_product` and an adjoint with `dagger`, both kept unevaluated until the compiler lowers them to matrices.
    Coefficients are ordinary expressions, so an operator is a sum of numbers or parameters times local operators,
    such as `J * operator_product(a, b)` rather than `J * a * b`.

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
        >>> from skcomplex.core.components.laws.utils import local_operator
        >>> local_operator(SR.symbol('q'), 'z')
        local_operator(q, z)

        A coupling of two subsystems with a strength `J` is a product of their operators.

        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.components.laws.utils import dagger, operator_product
        >>> a = local_operator(SR.symbol('a'), 'x')
        >>> b = local_operator(SR.symbol('b'), 'minus')
        >>> var('J') * operator_product(a, b) + dagger(b)
        J*operator_product(local_operator(a, x), local_operator(b, minus)) + dagger(local_operator(b, minus))
    """
    return LOCAL_OPERATOR(subsystem, SR.symbol(name) if isinstance(name, str) else name)


def has_operator(expression: object) -> bool:
    """Check whether an expression holds a local operator.

    A local operator makes it an operator rather than a scalar.

    Args:
        expression:
            A Sage expression, or any other value.

    Returns:
        has_operator:
            Whether a local operator appears in the expression.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, var
        >>> from skcomplex.core.components.laws.utils import local_operator
        >>> from skcomplex.core.components.laws.utils._local_operators import has_operator
        >>> has_operator(2 * local_operator(SR.symbol('q'), 'z')), has_operator(var('J'))
        (True, False)
    """
    return isinstance(expression, Expression) and expression.has(LOCAL_OPERATOR(SR.wild(0), SR.wild(1)))
