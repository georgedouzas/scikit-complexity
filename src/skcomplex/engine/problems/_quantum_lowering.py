"""Lowering of operators of quantum subsystems to sparse matrices on the whole space, with their coefficients."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import math
import operator
from collections.abc import Callable, Mapping

import numpy as np
import scipy.sparse as sp
from sage.all__sagemath_symbolics import SR, ZZ, Expression
from sage.symbolic.operators import add_vararg, mul_vararg

from ...base import BaseConfig
from ...core.laws.utils import dagger, has_operator, operator_product
from ...core.laws.utils._quantum import LOCAL_OPERATOR
from ...errors import ValidationError
from ...specs import LocalOperatorSpec, OperatorSpec, QuantumProblemSpec
from ...specs._subsystem_spec import _SubsystemSpec
from ...types import Array
from ._expression_printer import ExpressionPrinter
from ._quantum_kernel import QuantumKernel


class QuantumLowering(BaseConfig):
    """Lowering of operators of quantum subsystems to sparse matrices on the whole space and coefficients to functions.

    `run((expression, owner))` returns the terms of an operator, with constant terms folded and terms of equal
    coefficients summed. The owner, the description of the law or observable of the operator, names it in errors.

    Args:
        time:
            The time symbol.

        subsystems:
            The quantum subsystems by key, in tensor order.

        dimensions:
            The dimension of every subsystem, in tensor order.

        arguments:
            The index in the parameter vector of every parameter symbol.
    """

    def __init__(
        self,
        time: Expression,
        subsystems: Mapping[str, _SubsystemSpec],
        dimensions: tuple[int, ...],
        arguments: Mapping[Expression, int],
    ) -> None:
        self.time = time
        self.subsystems = subsystems
        self.dimensions = dimensions
        self.arguments = arguments

    def _run(self, inputs: tuple[object, str]) -> tuple[QuantumKernel, ...]:
        """Return the terms of an operator, with constant terms folded and terms of equal coefficients summed.

        Args:
            inputs:
                The Sage expression of the operator, built with `local_operator`, `operator_product`, and `dagger`, and
                the description of its law or observable.

        Returns:
            terms:
                The constant term first, if any, then one term per distinct coefficient.
        """
        expression, owner = inputs
        size = math.prod(self.dimensions)
        constant: list[sp.csr_array] = []
        grouped: dict[Expression, list[sp.csr_array]] = {}
        for key, value in self._operator(expression, owner).terms.items():
            coefficient = SR(value)
            matrix = self._matrix(key, owner)
            if coefficient.variables():
                grouped.setdefault(coefficient, []).append(matrix)
            else:
                constant.append(complex(coefficient) * matrix)
        terms = [QuantumKernel(QuantumProblemSpec._sum_matrices(constant, size), SR(1)).check()] if constant else []
        for coefficient, matrices in grouped.items():
            terms.append(
                QuantumKernel(
                    QuantumProblemSpec._sum_matrices(matrices, size),
                    coefficient,
                    self._function(coefficient, owner),
                ).check(),
            )
        return tuple(terms)

    def _operator(self, expression: object, owner: str) -> OperatorSpec:
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
            return self._operator_of(expression, subsystems)
        except ValidationError as error:
            error_msg = f'The operator of {owner} is invalid: {error}'
            raise ValidationError(error_msg) from error

    def _matrix(self, key: tuple[tuple[str, LocalOperatorSpec], ...], owner: str) -> sp.csr_array:
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
                error_msg = f'The operator of {owner} acts on {name}, which is not a quantum subsystem of the system.'
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

    def _function(self, coefficient: Expression, owner: str) -> Callable[[float, Array], complex]:
        """Return the function of a coefficient after checking that it reads parameters and time only.

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
                'parameter of the quantum system. Coefficients may read parameters and time only.'
            )
            raise ValidationError(error_msg)
        lowered = ExpressionPrinter(time=self.time, compiled=True).check().prepare(variables).run([coefficient])
        indices = np.asarray([self.arguments[item] for item in variables], dtype=np.intp)

        def evaluate(t: float, p: Array) -> complex:
            return complex(lowered(t, *p[indices])[0])

        return evaluate

    @staticmethod
    def _acted_on(operator: OperatorSpec) -> set[str]:
        """Return the keys of the subsystems any term of an operator acts on."""
        return {subsystem for key in operator.terms for subsystem, _ in key}

    @staticmethod
    def _local_of(
        expression: Expression,
        subsystems: Mapping[Expression, tuple[str, Mapping[str, LocalOperatorSpec]]],
    ) -> OperatorSpec:
        """Return the operator of a call `local_operator(subsystem, name)`."""
        subsystem, name = expression.operands()
        if subsystem not in subsystems:
            error_msg = f'{expression} acts on {subsystem}, which is not the state of a quantum element of the system.'
            raise ValidationError(error_msg)
        key, locals_ = subsystems[subsystem]
        if str(name) not in locals_:
            error_msg = f'{expression} names the operator {name}, but {key} has the operators {", ".join(locals_)}.'
            raise ValidationError(error_msg)
        return OperatorSpec({((key, locals_[str(name)]),): 1})

    @staticmethod
    def _product_of(factors: list[OperatorSpec], expression: Expression) -> OperatorSpec:
        """Return the product of the factors of a Sage product, which commute only on distinct subsystems."""
        acting = [QuantumLowering._acted_on(factor) for factor in factors if any(key for key in factor.terms)]
        if sum(len(item) for item in acting) != len(set().union(*acting)):
            error_msg = (
                f'{expression} multiplies operators that act on the same subsystem with `*`, which commutes. Write the '
                'product in its order with `operator_product`.'
            )
            raise ValidationError(error_msg)
        return functools.reduce(OperatorSpec.__mul__, factors, OperatorSpec({(): 1}))

    @staticmethod
    def _operator_of(
        expression: object,
        subsystems: Mapping[Expression, tuple[str, Mapping[str, LocalOperatorSpec]]],
    ) -> OperatorSpec:
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
            >>> from skcomplex.engine.problems._quantum_lowering import QuantumLowering
            >>> from skcomplex.specs import LocalOperatorSpec
            >>> q = SR.symbol('q')
            >>> z = LocalOperatorSpec('z', qutip.sigmaz(), hermitian=True)
            >>> QuantumLowering._operator_of(var('h') * local_operator(q, SR.symbol('z')), {q: ('q', {'z': z})}).text
            'h*q.z'
        """
        value = SR(expression)
        if not has_operator(value):
            return OperatorSpec({(): value})
        function, operands = value.operator(), value.operands()
        if function == LOCAL_OPERATOR:
            return QuantumLowering._local_of(value, subsystems)
        if function is operator.pow and operands[1] in ZZ and operands[1] > 0:
            function, operands = operator_product, [operands[0]] * int(operands[1])
        parts = [QuantumLowering._operator_of(item, subsystems) for item in operands]
        if function == dagger:
            return parts[0].adjoint()
        if function == operator_product:
            return functools.reduce(OperatorSpec.__mul__, parts)
        if function is add_vararg:
            return functools.reduce(OperatorSpec.__add__, parts)
        if function is mul_vararg:
            return QuantumLowering._product_of(parts, value)
        error_msg = f'{value} is not a sum of products of local operators with scalar coefficients.'
        raise ValidationError(error_msg)
