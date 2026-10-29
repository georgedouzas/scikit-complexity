"""Define the quantum problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from sage.all__sagemath_symbolics import SR, Expression

from ...specs._run_state_spec import RunStateSpec
from ...types import Array
from ._base_problem_spec import BaseProblemSpec
from ._quantum_kernel import QuantumKernel

REAL_TOLERANCE = 1e-10


@dataclass(frozen=True)
class QuantumProblemSpec(BaseProblemSpec):
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
        return QuantumProblemSpec._sum_matrices(
            (term.run(RunStateSpec(t, np.zeros(0), p)) * term.matrix for term in terms),
            size,
        )

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
            total += term.run(RunStateSpec(t, np.zeros(0), p)) * complex(value)
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
        return QuantumProblemSpec._matrix_at(self.hamiltonian, t, p, math.prod(self.dimensions))
