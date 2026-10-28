"""Hold operators lowered from the Sage operator expressions of quantum laws, as sums of products of local operators."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import qutip
import scipy.sparse as sp
from sage.all__sagemath_symbolics import SR, ZZ, Expression
from sage.symbolic.operators import add_vararg, mul_vararg

from ...core.errors import ValidationError
from ...core.laws.utils import dagger, has_operator, operator_product
from ...core.laws.utils._quantum import LOCAL_OPERATOR

ADJOINT_NAMES = {'plus': 'minus', 'minus': 'plus', 'annihilate': 'create', 'create': 'annihilate'}
ADJOINT_SUFFIX = '.dag'

type Factor = tuple[str, 'LocalOperator']
type Term = tuple[Factor, ...]


def _coefficient(value: object) -> Expression | None:
    """Return a number or a symbolic expression as a Sage expression, or `None` for any other value."""
    if isinstance(value, np.ndarray | Operator | str):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return SR(value)
    except (TypeError, ValueError):
        return None


def _added(terms: dict[Term, Expression], key: Term, coefficient: Expression) -> None:
    """Add a coefficient to the term of a key, in place."""
    terms[key] = terms[key] + coefficient if key in terms else coefficient


def sparse_matrix(matrix: Any) -> sp.csr_array:  # noqa: ANN401
    """Return a matrix, a QuTiP object, or an array as a complex sparse array."""
    if isinstance(matrix, qutip.Qobj):
        matrix = matrix.full()
    return sp.csr_array(np.asarray(matrix.toarray() if sp.issparse(matrix) else matrix, dtype=complex))


@dataclass(frozen=True)
class LocalOperator:
    """Operator that acts on the Hilbert space of one subsystem, identified by its name.

    Args:
        name:
            The name of the operator, unique among the operators of its subsystem.

        matrix:
            The matrix of the operator.

        hermitian:
            Whether the operator equals its adjoint.

    Examples:
        >>> import qutip
        >>> from skcomplex.engine.compile import LocalOperator
        >>> LocalOperator('plus', qutip.sigmap()).adjoint().name
        'minus'
    """

    name: str
    matrix: Any = field(default=None, compare=False, repr=False)
    hermitian: bool = field(default=False, compare=False)

    def adjoint(self) -> 'LocalOperator':
        """Return the adjoint of the operator.

        Returns:
            adjoint:
                The operator itself when it is Hermitian, or the operator with the conjugate transposed matrix.
        """
        if self.hermitian:
            return self
        if self.name in ADJOINT_NAMES:
            name = ADJOINT_NAMES[self.name]
        elif self.name.endswith(ADJOINT_SUFFIX):
            name = self.name.removesuffix(ADJOINT_SUFFIX)
        else:
            name = f'{self.name}{ADJOINT_SUFFIX}'
        matrix = sparse_matrix(self.matrix).conj().T.tocsr()
        return LocalOperator(name, matrix, hermitian=False)


class Operator:
    """Sum of products of local operators of subsystems, each product with a symbolic coefficient.

    The compiler lowers the Sage operator expressions of laws and observables to operators with `operator_of`. Operators
    combine by sums, products, scalar multiplication, and `adjoint`. A coefficient is a number or a
    Sage expression of parameters and time. A number added to an operator stands for that multiple of the identity.

    Args:
        terms:
            A mapping from a tuple of factors, each the key of a subsystem and a local operator, in product order, to
            the coefficient of the product. The empty tuple is the identity.

    Examples:
        >>> import qutip
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.engine.compile import LocalOperator, Operator
        >>> x = LocalOperator('x', qutip.sigmax(), hermitian=True)
        >>> a, b = Operator({(('a', x),): 1}), Operator({(('b', x),): 1})
        >>> (var('J') * a * b).text
        'J*a.x*b.x'
    """

    __array_ufunc__ = None

    def __init__(self, terms: Mapping[Term, object]) -> None:
        self.terms = terms

    def _items(self) -> dict[Term, Expression]:
        """Return the terms with Sage coefficients."""
        return {key: SR(value) for key, value in self.terms.items()}

    @staticmethod
    def _combined(terms: dict[Term, Expression]) -> 'Operator':
        """Return an operator from terms, without the terms whose coefficient is zero."""
        return Operator({key: value for key, value in terms.items() if not value.is_trivial_zero()})

    def _as_operator(self, other: object) -> 'Operator | None':
        """Return an operator, or a scalar as a multiple of the identity, or `None` for any other value."""
        if isinstance(other, Operator):
            return other
        coefficient = _coefficient(other)
        return None if coefficient is None else Operator({(): coefficient})

    def __add__(self, other: object) -> 'Operator':
        """Return the sum of the operator and an operator or a multiple of the identity."""
        operand = self._as_operator(other)
        if operand is None:
            return NotImplemented
        terms = self._items()
        for key, value in operand._items().items():
            _added(terms, key, value)
        return self._combined(terms)

    def __radd__(self, other: object) -> 'Operator':
        """Return the sum of a multiple of the identity and the operator."""
        return self.__add__(other)

    def __neg__(self) -> 'Operator':
        """Return the operator with every coefficient negated."""
        return Operator({key: -value for key, value in self._items().items()})

    def __sub__(self, other: object) -> 'Operator':
        """Return the difference of the operator and an operator or a multiple of the identity."""
        operand = self._as_operator(other)
        return NotImplemented if operand is None else self + (-operand)

    def __rsub__(self, other: object) -> 'Operator':
        """Return the difference of a multiple of the identity and the operator."""
        return (-self) + other

    def __mul__(self, other: object) -> 'Operator':
        """Return the product of the operator and an operator or a scalar, in this order."""
        operand = self._as_operator(other)
        if operand is None:
            return NotImplemented
        terms: dict[Term, Expression] = {}
        for left, first in self._items().items():
            for right, second in operand._items().items():
                _added(terms, left + right, first * second)
        return self._combined(terms)

    def __rmul__(self, other: object) -> 'Operator':
        """Return the product of a scalar and the operator."""
        operand = self._as_operator(other)
        return NotImplemented if operand is None else operand * self

    def __truediv__(self, other: object) -> 'Operator':
        """Return the operator divided by a scalar."""
        coefficient = _coefficient(other)
        return NotImplemented if coefficient is None else self * (1 / coefficient)

    @property
    def text(self) -> str:
        """Operator in operator notation, a sum of coefficients times products of local operators."""
        if not self.terms:
            return '0'
        parts = []
        for key, value in self._items().items():
            factors = '*'.join(f'{subsystem}.{local.name}' for subsystem, local in key)
            if not factors:
                parts.append(str(value))
            elif value == 1:
                parts.append(factors)
            else:
                text = str(value)
                parts.append(f'({text})*{factors}' if any(sign in text for sign in '+- ') else f'{text}*{factors}')
        return ' + '.join(parts)

    def adjoint(self) -> 'Operator':
        """Return the adjoint operator, with conjugate coefficients and reversed products of adjoint factors.

        Returns:
            adjoint:
                The adjoint operator.

        Examples:
            >>> import qutip
            >>> from skcomplex.engine.compile import LocalOperator, Operator
            >>> plus = LocalOperator('plus', qutip.sigmap())
            >>> (2j * Operator({(('q', plus),): 1})).adjoint().text
            '(-2j)*q.minus'
        """
        return Operator(
            {
                tuple((subsystem, local.adjoint()) for subsystem, local in reversed(key)): value.conjugate()
                for key, value in self._items().items()
            },
        )

    @property
    def subsystems(self) -> list[set[str]]:
        """Keys of the subsystems every term acts on, in term order."""
        return [{subsystem for subsystem, _ in key} for key in self.terms]


def _acted_on(operator: Operator) -> set[str]:
    """Return the keys of the subsystems any term of an operator acts on."""
    return {subsystem for key in operator.terms for subsystem, _ in key}


def _local_of(
    expression: Expression,
    subsystems: Mapping[Expression, tuple[str, Mapping[str, LocalOperator]]],
) -> Operator:
    """Return the operator of a call `local_operator(subsystem, name)`."""
    subsystem, name = expression.operands()
    if subsystem not in subsystems:
        error_msg = f'{expression} acts on {subsystem}, which is not the state of a quantum element of the system.'
        raise ValidationError(error_msg)
    key, locals_ = subsystems[subsystem]
    if str(name) not in locals_:
        error_msg = f'{expression} names the operator {name}, but {key} has the operators {", ".join(locals_)}.'
        raise ValidationError(error_msg)
    return Operator({((key, locals_[str(name)]),): 1})


def _product_of(factors: list[Operator], expression: Expression) -> Operator:
    """Return the product of the factors of a Sage product, which commute only on distinct subsystems."""
    acting = [_acted_on(factor) for factor in factors if any(key for key in factor.terms)]
    if sum(len(item) for item in acting) != len(set().union(*acting)):
        error_msg = (
            f'{expression} multiplies operators that act on the same subsystem with `*`, which commutes. Write the '
            'product in its order with `operator_product`.'
        )
        raise ValidationError(error_msg)
    return functools.reduce(Operator.__mul__, factors, Operator({(): 1}))


def operator_of(
    expression: object,
    subsystems: Mapping[Expression, tuple[str, Mapping[str, LocalOperator]]],
) -> Operator:
    """Return the operator of a Sage expression of local operators, products, and adjoints.

    Args:
        expression:
            A Sage expression built with `local_operator`, `operator_product`, and `dagger`, or a number or a scalar
            expression, which stands for that multiple of the identity.

        subsystems:
            A mapping from the state symbol of every quantum subsystem to its key and its local operators by name.

    Returns:
        operator:
            The operator.

    Raises:
        ValidationError:
            A local operator acts on an unknown subsystem or names an unknown operator, or a Sage product multiplies
            operators of the same subsystem.

    Examples:
        >>> import qutip
        >>> from sage.all__sagemath_symbolics import SR, var
        >>> from skcomplex.core.laws.utils import local_operator
        >>> from skcomplex.engine.compile import LocalOperator
        >>> from skcomplex.engine.compile._quantum_operators import operator_of
        >>> q = SR.symbol('q')
        >>> z = LocalOperator('z', qutip.sigmaz(), hermitian=True)
        >>> operator_of(var('h') * local_operator(q, SR.symbol('z')), {q: ('q', {'z': z})}).text
        'h*q.z'
    """
    value = SR(expression)
    if not has_operator(value):
        return Operator({(): value})
    function, operands = value.operator(), value.operands()
    if function == LOCAL_OPERATOR:
        return _local_of(value, subsystems)
    if function is operator.pow and operands[1] in ZZ and operands[1] > 0:
        function, operands = operator_product, [operands[0]] * int(operands[1])
    parts = [operator_of(item, subsystems) for item in operands]
    if function == dagger:
        return parts[0].adjoint()
    if function == operator_product:
        return functools.reduce(Operator.__mul__, parts)
    if function is add_vararg:
        return functools.reduce(Operator.__add__, parts)
    if function is mul_vararg:
        return _product_of(parts, value)
    error_msg = f'{value} is not a sum of products of local operators with scalar coefficients.'
    raise ValidationError(error_msg)
