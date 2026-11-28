"""Lowering of operators of subsystems to sparse matrices on the whole space."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import qutip
import scipy.sparse as sp
from sage.all__sagemath_symbolics import SR, ZZ, Expression
from sage.symbolic.operators import add_vararg, mul_vararg

from ...base import BaseConfig
from ...components.elements import Element
from ...components.laws.utils import dagger, operator_product
from ...components.laws.utils._local_operators import LOCAL_OPERATOR, has_operator
from ...errors import ValidationError
from ...types import Array
from ..types import HilbertSpaceProtocol
from ._expression_printer import ExpressionPrinter
from ._operator_kernel import OperatorKernel

NORM_TOLERANCE = 1e-8
ADJOINT_NAMES = {'plus': 'minus', 'minus': 'plus', 'annihilate': 'create', 'create': 'annihilate'}
ADJOINT_SUFFIX = '.dag'

type Factor = tuple[str, '_LocalOperator']

type Product = tuple[Factor, ...]


@dataclass(frozen=True)
class _LocalOperator:
    """A record of a named operator that acts on the space of one subsystem.

    It holds the name and the matrix of an operator of one subsystem, such as the raising operator `'plus'` of a qubit,
    and gives its adjoint, the lowering operator `'minus'`. Products of them make the operators of the whole system, for
    qubits, spins, and the modes of light.

    Read more in the user guide, under Extend the library.

    Args:
        name:
            The name of the operator, unique among the operators of its subsystem.

        matrix:
            The matrix of the operator.

        hermitian:
            Whether the operator equals its adjoint.

    See Also:
        _Operator: A sum of products of local operators.
    """

    name: str
    matrix: object = field(default=None, compare=False, repr=False)
    hermitian: bool = field(default=False, compare=False)

    def get_adjoint(self) -> '_LocalOperator':
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
        matrix = self._get_sparse_matrix(self.matrix).conj().T.tocsr()
        return _LocalOperator(name, matrix, hermitian=False)

    @staticmethod
    def _get_sparse_matrix(matrix: object) -> sp.csr_array:
        """Return a matrix, a QuTiP object, or an array as a complex sparse array."""
        if isinstance(matrix, qutip.Qobj):
            dense = matrix.full()
        elif isinstance(matrix, sp.sparray | sp.spmatrix):
            dense = matrix.toarray()
        else:
            dense = matrix
        return sp.csr_array(np.asarray(dense, dtype=complex))


@dataclass(frozen=True, eq=False)
class _Operator:
    """A record of a sum of products of local operators of subsystems.

    Each product of local operators has a coefficient, a number or a Sage expression of parameters and time, such as
    `J*a.x*b.x` for the coupling of two spins. Operators combine with `add`, `subtract`, `multiply`, `divide`, `negate`,
    and `adjoint`, a number stands for that multiple of the identity, and a product keeps its order. The compiler turns
    the operator expressions of laws and observables into it, for qubits, spin chains, and quantum optics.

    Read more in the user guide, under Extend the library.

    Args:
        terms:
            A mapping from a tuple of factors, each the key of a subsystem and a local operator, in product order, to
            the coefficient of the product. The empty tuple is the identity.

    See Also:
        _LocalOperator: An operator of one subsystem.
    """

    terms: Mapping[Product, object]

    def _get_items(self) -> dict[Product, Expression]:
        """Return the terms with Sage coefficients."""
        return {key: SR(value) for key, value in self.terms.items()}

    @staticmethod
    def _get_combined(terms: dict[Product, Expression]) -> '_Operator':
        """Return an operator from terms, without the terms whose coefficient is zero."""
        return _Operator({key: value for key, value in terms.items() if not value.is_trivial_zero()})

    def _convert_to_operator(self, other: object, operation: str) -> '_Operator':
        """Return an operator, or a scalar as a multiple of the identity.

        Args:
            other:
                The operand.

            operation:
                The name of the operation, for the error message.

        Returns:
            operator:
                The operand as an operator.

        Raises:
            ValidationError:
                The operand is neither an operator nor a scalar.
        """
        if isinstance(other, _Operator):
            return other
        return _Operator({(): self._get_scalar(other, operation)})

    def _get_scalar(self, other: object, operation: str) -> Expression:
        """Return a scalar as a Sage expression.

        Args:
            other:
                The operand.

            operation:
                The name of the operation, for the error message.

        Returns:
            coefficient:
                The operand as a Sage expression.

        Raises:
            ValidationError:
                The operand is not a number or a symbolic expression.
        """
        coefficient = self._get_coefficient(other)
        if coefficient is None:
            error_msg = (
                f'An operator cannot {operation} {other!r} of type {type(other).__name__}. The operand must be an '
                'operator, a number, or a Sage expression.'
            )
            raise ValidationError(error_msg)
        return coefficient

    def add(self, other: object) -> '_Operator':
        """Return the sum of the operator and an operator or a multiple of the identity.

        Args:
            other:
                An operator, or a number or a Sage expression, which stands for that multiple of the identity.

        Returns:
            sum:
                The sum.

        Raises:
            ValidationError:
                The operand is neither an operator nor a scalar.
        """
        terms = self._get_items()
        for key, value in self._convert_to_operator(other, 'add')._get_items().items():
            self._add_term(terms, key, value)
        return self._get_combined(terms)

    def negate(self) -> '_Operator':
        """Return the operator with every coefficient negated.

        Returns:
            negation:
                The negated operator.
        """
        return _Operator({key: -value for key, value in self._get_items().items()})

    def subtract(self, other: object) -> '_Operator':
        """Return the difference of the operator and an operator or a multiple of identity.

        Args:
            other:
                An operator, or a number or a Sage expression, which stands for that multiple of the identity.

        Returns:
            difference:
                The difference.

        Raises:
            ValidationError:
                The operand is neither an operator nor a scalar.
        """
        return self.add(self._convert_to_operator(other, 'subtract').negate())

    def multiply(self, other: object) -> '_Operator':
        """Return the product of the operator and an operator or a scalar, in this order.

        Args:
            other:
                The right factor, an operator, or a number or a Sage expression, which stands for that multiple of the
                identity.

        Returns:
            product:
                The product.

        Raises:
            ValidationError:
                The operand is neither an operator nor a scalar.
        """
        operand = self._convert_to_operator(other, 'multiply by')
        terms: dict[Product, Expression] = {}
        for left, first in self._get_items().items():
            for right, second in operand._get_items().items():
                self._add_term(terms, left + right, first * second)
        return self._get_combined(terms)

    def divide(self, other: object) -> '_Operator':
        """Return the operator divided by a scalar.

        Args:
            other:
                A number or a Sage expression.

        Returns:
            quotient:
                The quotient.

        Raises:
            ValidationError:
                The operand is not a scalar.
        """
        return self.multiply(1 / self._get_scalar(other, 'divide by'))

    @property
    def text(self) -> str:
        """_Operator in operator notation.

        It is a sum of coefficients times products of local operators.
        """
        if not self.terms:
            return '0'
        parts = []
        for key, value in self._get_items().items():
            factors = '*'.join(f'{subsystem}.{local.name}' for subsystem, local in key)
            if not factors:
                parts.append(str(value))
            elif value == 1:
                parts.append(factors)
            else:
                text = str(value)
                parts.append(f'({text})*{factors}' if any(sign in text for sign in '+- ') else f'{text}*{factors}')
        return ' + '.join(parts)

    def get_adjoint(self) -> '_Operator':
        """Return the adjoint operator.

        It has conjugate coefficients and reversed products of adjoint factors.

        Returns:
            adjoint:
                The adjoint operator.
        """
        return _Operator(
            {
                tuple((subsystem, local.get_adjoint()) for subsystem, local in reversed(key)): value.conjugate()
                for key, value in self._get_items().items()
            },
        )

    @property
    def subsystems(self) -> list[set[str]]:
        """Keys of the subsystems every term acts on, in term order."""
        return [{subsystem for subsystem, _ in key} for key in self.terms]

    @staticmethod
    def _get_coefficient(value: object) -> Expression | None:
        """Return a number or a symbolic expression as a Sage expression.

        It returns `None` for any other value.
        """
        if isinstance(value, np.ndarray | _Operator | str):
            return None
        if isinstance(value, np.generic):
            value = value.item()
        try:
            return SR(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _add_term(terms: dict[Product, Expression], key: Product, coefficient: Expression) -> None:
        """Add a coefficient to the term of a key, in place."""
        terms[key] = terms[key] + coefficient if key in terms else coefficient


@dataclass(frozen=True)
class _OperatorView:
    """A record of an element whose state is a vector of a Hilbert space.

    A Hilbert space is the space of the states of a quantum system. The compiler of operators reads this view to build
    the matrices of the evolution.

    Args:
        element:
            The element.

        state:
            The name of the state of the element in its Hilbert space.

        space:
            The Hilbert space of the element.
    """

    element: Element
    state: str
    space: HilbertSpaceProtocol

    @property
    def label(self) -> str:
        """Label of the element."""
        return self.element.label

    @property
    def dimension(self) -> int:
        """Dimension of the Hilbert space."""
        return self.space.dimension

    @property
    def display(self) -> str | None:
        """Operator whose expectation visuals show for a group.

        It is `None` for the probabilities of the basis.
        """
        return self.space.display

    def get_local_operators(self) -> dict[str, _LocalOperator]:
        """Return the local operators of the element by name.

        Returns:
            operators:
                A mapping from name to local operator.
        """
        return {
            name: _LocalOperator(name, _LocalOperator._get_sparse_matrix(matrix), hermitian=hermitian)
            for name, (matrix, hermitian) in self.space.operators.items()
        }

    def get_basis_index(self, label: object) -> int:
        """Return the index of the basis state of a label or an index.

        Args:
            label:
                The label of a basis state, or its index.

        Returns:
            index:
                The index of the basis state.

        Raises:
            ValidationError:
                The label names no basis state.
        """
        space = self.space
        if isinstance(label, str) and label in space.basis:
            return space.basis[label]
        if isinstance(label, int | np.integer) and not isinstance(label, bool) and 0 <= label < space.dimension:
            return int(label)
        labels = ', '.join(map(repr, space.basis)) or 'none'
        error_msg = (
            f'{type(self.element).__name__} {self.label!r}: {label!r} is no basis state. Use a label ({labels}) or an '
            f'index from 0 to {space.dimension - 1}.'
        )
        raise ValidationError(error_msg)

    def get_initial_vector(self) -> npt.NDArray[np.complex128]:
        """Return the initial state of the element as a normalised vector.

        Returns:
            vector:
                The initial state.

        Raises:
            ValidationError:
                The initial state names no basis state, or is a vector of the wrong size or norm.
        """
        given = getattr(self.element, 'values', None) or {}
        initial = given[self.state] if self.state in given else getattr(self.element, self.state)
        if isinstance(initial, str | int | np.integer):
            vector = np.zeros(self.dimension, dtype=complex)
            vector[self.get_basis_index(initial)] = 1.0
            return vector
        vector = np.asarray(initial, dtype=complex).ravel()
        norm = float(np.linalg.norm(vector))
        if vector.size != self.dimension or abs(norm - 1) > NORM_TOLERANCE:
            error_msg = (
                f'The initial vector of {self.label!r} has {vector.size} components and norm {norm:.6g}, but its '
                f'space has dimension {self.dimension} and states have norm 1. Normalise the vector.'
            )
            raise ValidationError(error_msg)
        return vector


@dataclass(frozen=True)
class _Subsystem:
    """A record of a subsystem of a tensor product, such as one qubit of a register.

    The compiler of operators builds one per operator element, and the lowering of operators reads them.

    Args:
        key:
            The key of the subsystem, the path of its element.

        element:
            The operator view of the element.

        index:
            The index of the subsystem in the tensor product.

        symbol:
            The state symbol of the subsystem.

        operators:
            The local operators of the element, by name.
    """

    key: str
    element: _OperatorView
    index: int
    symbol: Expression
    operators: Mapping[str, _LocalOperator]


class OperatorLowering(BaseConfig[None, tuple[object, str]]):
    """A lowering of the operators of subsystems to sparse matrices on the whole space.

    A quantum law names operators of single subsystems, such as the Pauli `z` of one qubit, and the engine needs them as
    matrices on the tensor product of all subsystems. `run((expression, owner))` turns an operator expression into
    operator kernels, each a sparse matrix with a coefficient compiled to a function, with constant terms folded and
    terms of equal coefficients summed. The owner, the law or observable of the operator, names it in errors. It builds
    the generators of qubits, spin chains, and atoms in a cavity.

    Read more in the user guide, under Extend the library.

    Args:
        time:
            The time symbol.

        subsystems:
            The subsystems by key, in tensor order.

        dimensions:
            The dimension of every subsystem, in tensor order.

        arguments:
            The index in the parameter vector of every parameter symbol.

    Attributes:
        kernels_:
            The terms of the operator of the last run: the constant term first, if any, then one term per distinct
            coefficient.

    See Also:
        OperatorKernel: One term that the lowering builds.

    Examples:
        >>> from skcomplex.core.systems import System
        >>> from skcomplex.systems.physics.common.quantum import QubitElement
        >>> problem = System(elements=[QubitElement('a')]).prepare().simulate(until=0).problem
        >>> type(problem.generator[0]).__name__
        'OperatorKernel'
    """

    kernels_: tuple[OperatorKernel, ...]

    def _prepare(self, context: None) -> None:
        """Derive nothing from the context, since an operator lowering prepares nothing."""

    def __init__(
        self,
        time: Expression,
        subsystems: Mapping[str, _Subsystem],
        dimensions: tuple[int, ...],
        arguments: Mapping[Expression, int],
    ) -> None:
        self.time = time
        self.subsystems = subsystems
        self.dimensions = dimensions
        self.arguments = arguments

    def _run(self, inputs: tuple[object, str] | None) -> None:
        """Set the terms of an operator.

        Constant terms are folded and terms of equal coefficients are summed.

        Args:
            inputs:
                The Sage expression of the operator, built with `local_operator`, `operator_product`, and `dagger`, and
                the description of its law or observable.

        Raises:
            ValidationError:
                No operator is passed.
        """
        if inputs is None:
            error_msg = 'An operator lowering runs on an operator and its owner. Pass them to `run`.'
            raise ValidationError(error_msg)
        expression, owner = inputs
        size = math.prod(self.dimensions)
        constant: list[sp.csr_array] = []
        grouped: dict[Expression, list[sp.csr_array]] = {}
        for key, value in self._get_operator(expression, owner).terms.items():
            coefficient = SR(value)
            matrix = self._get_matrix(key, owner)
            if coefficient.variables():
                grouped.setdefault(coefficient, []).append(matrix)
            else:
                constant.append(complex(coefficient) * matrix)
        terms = [OperatorKernel(OperatorKernel._sum_matrices(constant, size), SR(1)).prepare()] if constant else []
        for coefficient, matrices in grouped.items():
            terms.append(
                OperatorKernel(
                    OperatorKernel._sum_matrices(matrices, size),
                    coefficient,
                    self._get_function(coefficient, owner),
                ).prepare(),
            )
        self.kernels_ = tuple(terms)

    def _get_operator(self, expression: object, owner: str) -> _Operator:
        """Return the operator of a Sage expression of local operators of the subsystems.

        Args:
            expression:
                The expression, built with `local_operator`, `operator_product`, and `dagger`.

            owner:
                The description of the law or observable, for error messages.

        Returns:
            operator:
                The operator.

        Raises:
            ValidationError:
                The expression is not an operator of the subsystems.
        """
        subsystems = {item.symbol: (item.key, item.operators) for item in self.subsystems.values()}
        try:
            return self._get_operator_of(expression, subsystems)
        except ValidationError as error:
            error_msg = f'The operator of {owner} is invalid: {error}'
            raise ValidationError(error_msg) from error

    def _get_matrix(self, key: tuple[tuple[str, _LocalOperator], ...], owner: str) -> sp.csr_array:
        """Return the matrix of a product of local operators on the whole space.

        Args:
            key:
                The factors of the product, each the key of a subsystem and a local operator.

            owner:
                The description of the law or observable, for error messages.

        Returns:
            matrix:
                The sparse matrix.

        Raises:
            ValidationError:
                A factor acts on a subsystem that is not part of the system.
        """
        products: dict[int, sp.csr_array] = {}
        for name, local in key:
            if name not in self.subsystems:
                error_msg = f'The operator of {owner} acts on {name}, which is not a subsystem of the system.'
                raise ValidationError(error_msg)
            subsystem = self.subsystems[name]
            matrix = sp.csr_array(local.matrix)
            previous = products.get(subsystem.index)
            products[subsystem.index] = matrix if previous is None else sp.csr_array(previous @ matrix)
        blocks: list[sp.csr_array] = []
        pending = 1
        for index, size in enumerate(self.dimensions):
            if index not in products:
                pending *= size
                continue
            if pending > 1:
                blocks.append(sp.identity(pending, dtype=complex, format='csr'))
            blocks.append(products[index])
            pending = 1
        if pending > 1 or not blocks:
            blocks.append(sp.identity(pending, dtype=complex, format='csr'))
        return sp.csr_array(functools.reduce(lambda left, right: sp.kron(left, right, format='csr'), blocks))

    def _get_function(self, coefficient: Expression, owner: str) -> Callable[[float, Array], complex]:
        """Return the function of a coefficient.

        The coefficient is first checked to read parameters and time only.

        Args:
            coefficient:
                The symbolic coefficient.

            owner:
                The description of the law or observable, for error messages.

        Returns:
            function:
                The function `c(t, p)`.

        Raises:
            ValidationError:
                The coefficient uses a symbol that is neither a parameter nor time.
        """
        variables = sorted(set(coefficient.variables()) - {self.time}, key=str)
        unknown = [item for item in variables if item not in self.arguments]
        if unknown:
            error_msg = (
                f'The operator of {owner} has a coefficient that uses {", ".join(map(str, unknown))}, which is not a '
                'parameter of the operator system. Coefficients may read parameters and time only.'
            )
            raise ValidationError(error_msg)
        printer = ExpressionPrinter(time=self.time, compiled=True).prepare(variables)
        lowered = ExpressionPrinter._get_function_of(printer.run([coefficient]).code_)
        indices = np.asarray([self.arguments[item] for item in variables], dtype=np.intp)

        def evaluate(t: float, p: Array) -> complex:
            return OperatorKernel._get_coefficient_of(ExpressionPrinter._get_values_of(lowered(t, *p[indices]))[0])

        return evaluate

    @staticmethod
    def _get_terms_of(result: object) -> tuple[OperatorKernel, ...]:
        """Return the terms that `run` of an operator lowering returned.

        Args:
            result:
                What `run` returned.

        Returns:
            terms:
                The terms of the operator.

        Raises:
            TypeError:
                The result is not a tuple of term kernels.
        """
        if not isinstance(result, tuple) or not all(isinstance(item, OperatorKernel) for item in result):
            error_msg = f'An operator lowering returns a tuple of term kernels, not a {type(result).__name__}.'
            raise TypeError(error_msg)
        return result

    @staticmethod
    def _get_acted_on(operator: _Operator) -> set[str]:
        """Return the keys of the subsystems any term of an operator acts on."""
        return {subsystem for key in operator.terms for subsystem, _ in key}

    @staticmethod
    def _get_local_of(
        expression: Expression,
        subsystems: Mapping[Expression, tuple[str, Mapping[str, _LocalOperator]]],
    ) -> _Operator:
        """Return the operator of a call `local_operator(subsystem, name)`."""
        subsystem, name = expression.operands()
        if subsystem not in subsystems:
            error_msg = (
                f'{expression} acts on {subsystem}, which is not the state of an operator element of the system.'
            )
            raise ValidationError(error_msg)
        key, locals_ = subsystems[subsystem]
        if str(name) not in locals_:
            error_msg = f'{expression} names the operator {name}, but {key} has the operators {", ".join(locals_)}.'
            raise ValidationError(error_msg)
        return _Operator({((key, locals_[str(name)]),): 1})

    @staticmethod
    def _get_product_of(factors: list[_Operator], expression: Expression) -> _Operator:
        """Return the product of the factors of a Sage product.

        The factors commute only on distinct subsystems.
        """
        acting = [OperatorLowering._get_acted_on(factor) for factor in factors if any(key for key in factor.terms)]
        if sum(len(item) for item in acting) != len(set().union(*acting)):
            error_msg = (
                f'{expression} multiplies operators that act on the same subsystem with `*`, which commutes. Write the '
                'product in its order with `operator_product`.'
            )
            raise ValidationError(error_msg)
        return functools.reduce(_Operator.multiply, factors, _Operator({(): 1}))

    @staticmethod
    def _get_operator_of(
        expression: object,
        subsystems: Mapping[Expression, tuple[str, Mapping[str, _LocalOperator]]],
    ) -> _Operator:
        """Return the operator of a Sage expression.

        The expression is built of local operators, products, and adjoints.

        Args:
            expression:
                A Sage expression built with `local_operator`, `operator_product`, and `dagger`, or a number or a scalar
                expression, which stands for that multiple of the identity.

            subsystems:
                A mapping from the state symbol of every subsystem to its key and its local operators by name.

        Returns:
            operator:
                The operator.

        Raises:
            ValidationError:
                A local operator acts on an unknown subsystem or names an unknown operator, or a Sage product multiplies
                operators of the same subsystem.
        """
        value = SR(expression)
        if not has_operator(value):
            return _Operator({(): value})
        function, operands = value.operator(), value.operands()
        if function == LOCAL_OPERATOR:
            return OperatorLowering._get_local_of(value, subsystems)
        if function is operator.pow and operands[1] in ZZ and operands[1] > 0:
            function, operands = operator_product, [operands[0]] * int(operands[1])
        parts = [OperatorLowering._get_operator_of(item, subsystems) for item in operands]
        if function == dagger:
            return parts[0].get_adjoint()
        if function == operator_product:
            return functools.reduce(_Operator.multiply, parts)
        if function is add_vararg:
            return functools.reduce(_Operator.add, parts)
        if function is mul_vararg:
            return OperatorLowering._get_product_of(parts, value)
        error_msg = f'{value} is not a sum of products of local operators with scalar coefficients.'
        raise ValidationError(error_msg)
