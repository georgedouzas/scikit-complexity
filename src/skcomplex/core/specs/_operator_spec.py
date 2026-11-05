"""Define the operator spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from sage.all__sagemath_symbolics import SR, Expression

from ._base_spec import BaseSpec
from ._local_operator_spec import LocalOperatorSpec

type Factor = tuple[str, 'LocalOperatorSpec']


type Product = tuple[Factor, ...]


@dataclass(frozen=True, eq=False)
class OperatorSpec(BaseSpec):
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
        >>> from skcomplex.core.specs import LocalOperatorSpec, OperatorSpec
        >>> x = LocalOperatorSpec('x', qutip.sigmax(), hermitian=True)
        >>> a, b = OperatorSpec({(('a', x),): 1}), OperatorSpec({(('b', x),): 1})
        >>> (var('J') * a * b).text
        'J*a.x*b.x'
    """

    __array_ufunc__ = None

    terms: Mapping[Product, object]

    def _items(self) -> dict[Product, Expression]:
        """Return the terms with Sage coefficients."""
        return {key: SR(value) for key, value in self.terms.items()}

    @staticmethod
    def _combined(terms: dict[Product, Expression]) -> 'OperatorSpec':
        """Return an operator from terms, without the terms whose coefficient is zero."""
        return OperatorSpec({key: value for key, value in terms.items() if not value.is_trivial_zero()})

    def _as_operator(self, other: object) -> 'OperatorSpec | None':
        """Return an operator, or a scalar as a multiple of the identity, or `None` for any other value."""
        if isinstance(other, OperatorSpec):
            return other
        coefficient = self._coefficient(other)
        return None if coefficient is None else OperatorSpec({(): coefficient})

    def __add__(self, other: object) -> 'OperatorSpec':
        """Return the sum of the operator and an operator or a multiple of the identity."""
        operand = self._as_operator(other)
        if operand is None:
            return NotImplemented
        terms = self._items()
        for key, value in operand._items().items():
            self._added(terms, key, value)
        return self._combined(terms)

    def __radd__(self, other: object) -> 'OperatorSpec':
        """Return the sum of a multiple of the identity and the operator."""
        return self + other

    def __neg__(self) -> 'OperatorSpec':
        """Return the operator with every coefficient negated."""
        return OperatorSpec({key: -value for key, value in self._items().items()})

    def __sub__(self, other: object) -> 'OperatorSpec':
        """Return the difference of the operator and an operator or a multiple of the identity."""
        operand = self._as_operator(other)
        return NotImplemented if operand is None else self + (-operand)

    def __rsub__(self, other: object) -> 'OperatorSpec':
        """Return the difference of a multiple of the identity and the operator."""
        return (-self) + other

    def __mul__(self, other: object) -> 'OperatorSpec':
        """Return the product of the operator and an operator or a scalar, in this order."""
        operand = self._as_operator(other)
        if operand is None:
            return NotImplemented
        terms: dict[Product, Expression] = {}
        for left, first in self._items().items():
            for right, second in operand._items().items():
                self._added(terms, left + right, first * second)
        return self._combined(terms)

    def __rmul__(self, other: object) -> 'OperatorSpec':
        """Return the product of a scalar and the operator."""
        operand = self._as_operator(other)
        return NotImplemented if operand is None else operand * self

    def __truediv__(self, other: object) -> 'OperatorSpec':
        """Return the operator divided by a scalar."""
        coefficient = self._coefficient(other)
        return NotImplemented if coefficient is None else self * (1 / coefficient)

    @property
    def text(self) -> str:
        """OperatorSpec in operator notation, a sum of coefficients times products of local operators."""
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

    def adjoint(self) -> 'OperatorSpec':
        """Return the adjoint operator, with conjugate coefficients and reversed products of adjoint factors.

        Returns:
            adjoint:
                The adjoint operator.

        Examples:
            >>> import qutip
            >>> from skcomplex.core.specs import LocalOperatorSpec, OperatorSpec
            >>> plus = LocalOperatorSpec('plus', qutip.sigmap())
            >>> (2j * OperatorSpec({(('q', plus),): 1})).adjoint().text
            '(-2j)*q.minus'
        """
        return OperatorSpec(
            {
                tuple((subsystem, local.adjoint()) for subsystem, local in reversed(key)): value.conjugate()
                for key, value in self._items().items()
            },
        )

    @property
    def subsystems(self) -> list[set[str]]:
        """Keys of the subsystems every term acts on, in term order."""
        return [{subsystem for subsystem, _ in key} for key in self.terms]

    @staticmethod
    def _coefficient(value: object) -> Expression | None:
        """Return a number or a symbolic expression as a Sage expression, or `None` for any other value."""
        if isinstance(value, np.ndarray | OperatorSpec | str):
            return None
        if isinstance(value, np.generic):
            value = value.item()
        try:
            return SR(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _added(terms: dict[Product, Expression], key: Product, coefficient: Expression) -> None:
        """Add a coefficient to the term of a key, in place."""
        terms[key] = terms[key] + coefficient if key in terms else coefficient
