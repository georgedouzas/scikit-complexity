"""Problem of the quantum form, whose state is a pure state or a density matrix stored as complex pairs."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from sage.all__sagemath_symbolics import SR, Expression

from ...core.laws.utils import has_operator, local_operator
from ...errors import ValidationError
from ...specs import ObservableSpec
from ...types import Array, Getter
from ._base import BaseProblem
from ._quantum_kernel import QuantumKernel
from ._reduction_kernel import ReductionKernel
from ._state import State

REAL_TOLERANCE = 1e-10


@dataclass(frozen=True)
class QuantumProblem(BaseProblem):
    """Problem of the quantum form, whose state is a pure state or a density matrix stored as complex pairs.

    Args:
        dimensions:
            The Hilbert space dimensions of the subsystems, in tensor product order.

        subsystems:
            The keys of the subsystems, such as `chain[3]`, in tensor product order.

        hamiltonian:
            The terms of the Hamiltonian.

        dissipators:
            The terms of every collapse operator, each with the square root of its rate in the coefficients.

        pure:
            Whether the state is a pure state vector rather than a density matrix.

        namespaces:
            The namespace of every system, by its path, which observables receive.

        members:
            The namespaces of the members of every quantum element or group, by path.

        lowering:
            The lowering of operators to matrices and functions.

        hamiltonian_operator:
            The sum of the operators of the generator laws, a Sage expression of local operators.

        cache:
            The matrices of local operators computed so far.
    """

    dimensions: tuple[int, ...] = ()
    subsystems: tuple[str, ...] = ()
    hamiltonian: tuple[QuantumKernel, ...] = field(default=(), repr=False)
    dissipators: tuple[tuple[QuantumKernel, ...], ...] = field(default=(), repr=False)
    pure: bool = True
    namespaces: Mapping[str, SimpleNamespace] = field(default_factory=dict, repr=False)
    members: Mapping[str, tuple[SimpleNamespace, ...]] = field(default_factory=dict, repr=False)
    lowering: Any = field(default=None, repr=False)
    hamiltonian_operator: Expression = field(default=SR(0), repr=False)
    cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def observable_getters(self, observables: Iterable[tuple[str, ObservableSpec]]) -> dict[str, Getter]:
        """Return the getters of the observables, with the energy, and the purity when the system is open.

        An observable gives an operator, a Sage expression of local operators, as a function of the namespace of its
        system, of its element, or of every member of its group, or names a local operator of its element or of every
        member of its group with `variable`, such as `ObservableSpec('z', element='q', variable='z')`. Its value is the
        expectation of the operator, reduced over the members of a group.

        Args:
            observables:
                The path of the declaring system and the observable, for every observable of the system tree.

        Returns:
            getters:
                The getters by observable path.

        Raises:
            ValidationError:
                Two observables share a path, or an observable does not give an operator.
        """
        getters: dict[str, Getter] = {}
        for scope, observable in observables:
            label = observable.label if observable.element is None else f'{observable.element}.{observable.label}'
            path = f'{scope}/{label}' if scope else label
            if path in getters:
                error_msg = f'Two observables have the path {path}. Give observables of one system distinct labels.'
                raise ValidationError(error_msg)
            if observable.group is not None:
                group = f'{scope}/{observable.group}' if scope else observable.group
                getters[path] = self._member_getter(path, observable, group)
                continue
            if observable.element is not None and observable.variable is not None:
                element = f'{scope}/{observable.element}' if scope else observable.element
                getters[path] = self._expectation_getter(self._local_terms(element, observable.variable)[0])
                continue
            expression = observable.expression
            value = expression(self.namespaces[scope]) if callable(expression) else expression
            getters[path] = self._expectation_getter(
                self.lowering.terms(self._operator_of(path, value), path),
            )
        getters.setdefault('energy', self._expectation_getter(self.hamiltonian))
        if not self.pure:
            getters.setdefault('purity', self._purity_getter())
        return getters

    def _local_terms(self, path: str, name: str) -> list[tuple[QuantumKernel, ...]]:
        """Return the terms of the local operator of a name of every member of a quantum element or group."""
        subsystems = [
            item for key, item in self.lowering.subsystems.items() if key == path or key.startswith(f'{path}[')
        ]
        if not subsystems:
            error_msg = f'Observable of {path} reads {path}, which is not a quantum element or group of the system.'
            raise ValidationError(error_msg)
        return [self.lowering.terms(local_operator(item.symbol, SR.symbol(name)), path) for item in subsystems]

    def _member_getter(self, path: str, observable: ObservableSpec, group: str) -> Getter:
        """Return the getter of a quantum observable evaluated member by member and reduced.

        The observable gives the operator of a member as a function of its namespace, or names a local operator of every
        member with `variable`.
        """
        if group not in self.members:
            error_msg = f'Observable {path} reduces {group}, which is not a quantum group of the system.'
            raise ValidationError(error_msg)
        expression = observable.expression
        if observable.variable is not None:
            terms = self._local_terms(group, observable.variable)
        elif callable(expression):
            members = self.members[group]
            terms = [self.lowering.terms(self._operator_of(path, expression(member)), path) for member in members]
        else:
            error_msg = f'Observable {path} of a quantum group needs a function of a member namespace or a variable.'
            raise ValidationError(error_msg)

        def get(t: float, x: Array, p: Array, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
            del done
            values = np.asarray([self._expect(item, t, x, p) for item in terms])
            return ReductionKernel._reduce_values(values, observable.reduce)

        return get

    @staticmethod
    def _operator_of(path: str, value: object) -> Expression:
        """Return the operator an observable expression gives, after checking that it is one."""
        if not has_operator(value):
            error_msg = f'Observable {path} of a quantum system gives {value!r}, but it must give an operator.'
            raise ValidationError(error_msg)
        return value

    def _expectation_getter(self, terms: tuple[QuantumKernel, ...]) -> Getter:
        """Return the getter of the expectation value of an operator."""

        def get(t: float, x: Array, p: Array, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
            del done
            return self._expect(terms, t, x, p)

        return get

    def _purity_getter(self) -> Getter:
        """Return the getter of the purity `Tr(rho^2)` of the state."""

        def get(t: float, x: Array, p: Array, done: Mapping[str, Any]) -> Any:  # noqa: ANN401
            del t, p, done
            state = self._state_of(x)
            return 1.0 if self.pure else float(np.sum(np.abs(state) ** 2))

        return get

    @staticmethod
    def _sum_matrices(matrices: Iterable[sp.csr_array], size: int) -> sp.csr_array:
        """Return the sum of sparse matrices, the zero matrix without any."""
        total = sp.csr_array((size, size), dtype=complex)
        for matrix in matrices:
            total = total + matrix
        return sp.csr_array(total)

    @staticmethod
    def _matrix_at(terms: Iterable[QuantumKernel], t: float, p: Array, size: int) -> sp.csr_array:
        """Return the sum of the terms at a time and a parameter vector."""
        return QuantumProblem._sum_matrices((term.run(State(t, np.zeros(0), p)) * term.matrix for term in terms), size)

    def _state_of(self, x: Array) -> npt.NDArray[np.complex128]:
        """Return the complex state of a quantum problem, a vector or a density matrix, from a state vector.

        Args:
            x:
                The state vector.

        Returns:
            state:
                The pure state of shape `(dimension,)` or the density matrix of shape `(dimension, dimension)`.
        """
        return np.asarray(self.states.read(x, self.states.names[0]))

    def _expect(self, terms: Iterable[QuantumKernel], t: float, x: Array, p: Array) -> float | complex:
        """Return the expectation value of an operator at a state of a quantum problem.

        Args:
            terms:
                The terms of the operator.

            t:
                The time.

            x:
                The state vector.

            p:
                The parameter vector.

        Returns:
            value:
                `<psi|O|psi>` or `Tr(rho O)`, a float when its imaginary part vanishes.
        """
        state = self._state_of(x)
        total = 0j
        for term in terms:
            value = np.vdot(state, term.matrix @ state) if self.pure else term.matrix.multiply(state.T).sum()
            total += term.run(State(t, np.zeros(0), p)) * complex(value)
        if abs(total.imag) <= REAL_TOLERANCE * (1 + abs(total.real)):
            return float(total.real)
        return total

    def _hamiltonian_at(self, t: float, p: Array) -> sp.csr_array:
        """Return the Hamiltonian matrix of a quantum problem at a time and a parameter vector.

        Args:
            t:
                The time.

            p:
                The parameter vector.

        Returns:
            matrix:
                The sparse Hamiltonian.
        """
        return QuantumProblem._matrix_at(self.hamiltonian, t, p, math.prod(self.dimensions))
