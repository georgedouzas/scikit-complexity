"""Define the operators that act on one subsystem of a tensor product."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import SR, Expression

from ._operators import _OperatorFunction

LOCAL_OPERATOR = _OperatorFunction('local_operator', nargs=2)


operator_product = _OperatorFunction('operator_product')
dagger = _OperatorFunction('dagger', nargs=1)


def local_operator(subsystem: object, name: object) -> Expression:
    """Return a named operator that acts on one subsystem alone.

    The name picks an operator of the Hilbert space of the subsystem, such as `local_operator(q, 'z')` for the Pauli `Z`
    of the qubit `q`. Operators do not commute, so a product is written `operator_product(a, b)` and an adjoint
    `dagger(a)`, both kept as they are until the compiler turns them into matrices. A coefficient is an ordinary
    expression, so a coupling of strength `J` is `J * operator_product(a, b)`. It builds the Hamiltonian of a spin
    chain, of a cavity holding light and an atom, and of a register of qubits.

    Read more in the user guide, under Laws.

    Args:
        subsystem:
            The symbol of the state of the subsystem, such as `namespace.psi`.

        name:
            The name of the operator in the Hilbert space, such as `'z'`, or its symbol.

    Returns:
        operator:
            The unevaluated local operator.

    See Also:
        has_operator: Checks whether an expression holds a local operator.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR
        >>> from skcomplex.core.components.laws.utils import local_operator
        >>> local_operator(SR.symbol('q'), 'z')
        local_operator(q, z)
    """
    return LOCAL_OPERATOR(subsystem, SR.symbol(name) if isinstance(name, str) else name)


def has_operator(expression: object) -> bool:
    """Check whether an expression holds a local operator.

    An expression that holds a local operator is an operator of a quantum system rather than a number, such as
    `2 * local_operator(q, 'z')`, while `J` alone is a number. The compiler reads it to tell a Hamiltonian from an
    ordinary rate. It sorts the terms of a spin chain, a cavity, and a register of qubits.

    Read more in the user guide, under Extend the library.

    Args:
        expression:
            A Sage expression, or any other value.

    Returns:
        has_operator:
            Whether a local operator appears in the expression.

    See Also:
        local_operator: Builds the operators it looks for.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, var
        >>> from skcomplex.core.components.laws.utils import local_operator
        >>> from skcomplex.core.components.laws.utils._local_operators import has_operator
        >>> has_operator(2 * local_operator(SR.symbol('q'), 'z')), has_operator(var('J'))
        (True, False)
    """
    return isinstance(expression, Expression) and expression.has(LOCAL_OPERATOR(SR.wild(0), SR.wild(1)))
