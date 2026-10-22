"""Define operators built from local operators of quantum subsystems, and the Hamiltonian and dissipator laws."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import qutip
import scipy.sparse as sp
from sage.all__sagemath_symbolics import SR

from ..base import DomainKind, Expression, LawKind, Requires, TimeKind, ValidationError

ADJOINT_NAMES = {'plus': 'minus', 'minus': 'plus', 'annihilate': 'create', 'create': 'annihilate'}
ADJOINT_SUFFIX = '.dag'

type Factor = tuple[str, 'LocalOperator']
type Term = tuple[Factor, ...]


def sparse_matrix(matrix: Any) -> sp.csr_array:  # noqa: ANN401
    """Return a matrix, a QuTiP object, or an array as a complex sparse array."""
    if isinstance(matrix, qutip.Qobj):
        matrix = matrix.full()
    return sp.csr_array(np.asarray(matrix.toarray() if sp.issparse(matrix) else matrix, dtype=complex))


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


@dataclass(frozen=True)
class LocalOperator:
    """Operator that acts on the Hilbert space of one subsystem, identified by its name.

    Args:
        name:
            The name of the operator, unique among the operators of its subsystem.

        matrix:
            The matrix of the operator, or `None` for an operator of the space that the compiler builds, such as the
            Laplacian of a lattice.

        hermitian:
            Whether the operator equals its adjoint.

    Examples:
        >>> import qutip
        >>> from skcomplex.core.model import LocalOperator
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
        matrix = None if self.matrix is None else sparse_matrix(self.matrix).conj().T.tocsr()
        return LocalOperator(name, matrix, hermitian=False)


class Operator:
    """Sum of products of local operators of subsystems, each product with a symbolic coefficient.

    Operators are built from the local operators of the namespaces that laws and observables receive, such as `s.x` or
    `a.annihilate`, and combine by sums, products, scalar multiplication, and `adjoint`. A coefficient is a number or a
    Sage expression of parameters and time. A number added to an operator stands for that multiple of the identity.

    Args:
        terms:
            A mapping from a tuple of factors, each the key of a subsystem and a local operator, in product order, to
            the coefficient of the product. The empty tuple is the identity.

    Examples:
        >>> import qutip
        >>> from skcomplex.core.base import symbol
        >>> from skcomplex.core.model import LocalOperator, Operator
        >>> x = LocalOperator('x', qutip.sigmax(), hermitian=True)
        >>> a, b = Operator({(('a', x),): 1}), Operator({(('b', x),): 1})
        >>> symbol('J') * a * b
        J*a.x*b.x
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

    def __repr__(self) -> str:
        """Return the operator in operator notation, a sum of coefficients times products of local operators."""
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
            >>> from skcomplex.core.model import LocalOperator, Operator
            >>> plus = LocalOperator('plus', qutip.sigmap())
            >>> (2j * Operator({(('q', plus),): 1})).adjoint()
            (-2j)*q.minus
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


@dataclass(frozen=True)
class Hamiltonian:
    """Law that adds an operator to the Hamiltonian of a quantum system, with `ħ = 1`.

    Args:
        operator:
            The Hermitian operator.

    Raises:
        ValidationError:
            The operator is not an `Operator`.

    Examples:
        >>> import qutip
        >>> from skcomplex.core.model import Hamiltonian, LocalOperator, Operator
        >>> z = LocalOperator('z', qutip.sigmaz(), hermitian=True)
        >>> Hamiltonian(Operator({(('q', z),): 0.5}))
        Hamiltonian(operator=0.5*q.z)
    """

    law_kind: ClassVar[LawKind] = LawKind.GENERATOR
    requires: ClassVar[Requires] = Requires(time={TimeKind.CONTINUOUS}, domains={DomainKind.HILBERT})

    operator: Operator

    def __post_init__(self) -> None:
        """Check that the law holds an operator."""
        if not isinstance(self.operator, Operator):
            error_msg = (
                f'A Hamiltonian takes an operator built from local operators, such as `s.z`, but got {self.operator!r}.'
            )
            raise ValidationError(error_msg)


@dataclass(frozen=True)
class Dissipator:
    """Law that adds a Lindblad dissipator with a collapse operator and a rate, which makes a quantum system open.

    The master equation gains `rate * (L rho L^† - (L^† L rho + rho L^† L) / 2)`.

    Args:
        operator:
            The collapse operator `L`.

        rate:
            The non-negative rate, a number or a Sage expression of parameters and time.

    Raises:
        ValidationError:
            The operator is not an `Operator`, or the rate is not a number or an expression.

    Examples:
        >>> import qutip
        >>> from skcomplex.core.model import Dissipator, LocalOperator, Operator
        >>> minus = LocalOperator('minus', qutip.sigmam())
        >>> Dissipator(Operator({(('q', minus),): 1}), 0.1).rate
        0.1
    """

    law_kind: ClassVar[LawKind] = LawKind.DISSIPATOR
    requires: ClassVar[Requires] = Requires(time={TimeKind.CONTINUOUS}, domains={DomainKind.HILBERT})

    operator: Operator
    rate: object = 1.0

    def __post_init__(self) -> None:
        """Check the collapse operator and the rate."""
        if not isinstance(self.operator, Operator):
            error_msg = f'A Dissipator takes an operator, such as `s.minus`, but got {self.operator!r}.'
            raise ValidationError(error_msg)
        if _coefficient(self.rate) is None:
            error_msg = f'The rate of a Dissipator is a number or a symbolic expression, but got {self.rate!r}.'
            raise ValidationError(error_msg)
