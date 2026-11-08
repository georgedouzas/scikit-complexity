"""Define the operator problem spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from sage.all__sagemath_symbolics import SR, Expression

from ...specs._run_state_spec import RunStateSpec
from ...types import Array
from ._base_problem_spec import BaseProblemSpec
from ._operator_kernel import OperatorKernel
from ._operator_lowering import OperatorLowering

REAL_TOLERANCE = 1e-10


@dataclass(frozen=True)
class OperatorProblemSpec(BaseProblemSpec):
    """Problem of the operator form.

    Its state is a pure state or a density matrix stored as complex pairs.

    Args:
        dimensions:
            The Hilbert space dimensions of the subsystems, in tensor product order.

        subsystems:
            The keys of the subsystems, such as `chain[3]`, in tensor product order.

        generator:
            The terms of the generator.

        dissipators:
            The terms of every collapse operator, each with the square root of its rate in the coefficients.

        pure:
            Whether the state is a pure state vector rather than a density matrix.

        namespaces:
            The namespace of every system, by its path, which observables receive.

        members:
            The namespaces of the members of every operator element or group, by path.

        lowering:
            The lowering of operators to matrices and functions.

        generator_operator:
            The sum of the operators of the generator laws, a Sage expression of local operators.

        cache:
            The matrices of local operators computed so far.
    """

    dimensions: tuple[int, ...] = ()
    subsystems: tuple[str, ...] = ()
    generator: tuple[OperatorKernel, ...] = field(default=(), repr=False)
    dissipators: tuple[tuple[OperatorKernel, ...], ...] = field(default=(), repr=False)
    pure: bool = True
    namespaces: Mapping[str, SimpleNamespace] = field(default_factory=dict, repr=False)
    members: Mapping[str, tuple[SimpleNamespace, ...]] = field(default_factory=dict, repr=False)
    lowering: OperatorLowering | None = field(default=None, repr=False)
    generator_operator: Expression = field(default=SR(0), repr=False)
    cache: dict[str, list[sp.csr_array]] = field(default_factory=dict, repr=False, compare=False)

    @staticmethod
    def _matrix_at(terms: Iterable[OperatorKernel], t: float, p: Array, size: int) -> sp.csr_array:
        """Return the sum of the terms at a time and a parameter vector."""
        return OperatorKernel._sum_matrices(
            (term.run(RunStateSpec(t, np.zeros(0), p)).values_ * term.matrix for term in terms),
            size,
        )

    def _state_of(self, x: Array) -> npt.NDArray[np.complex128]:
        """Return the complex state of an operator problem from a state vector.

        The complex state is a vector or a density matrix.

        Args:
            x:
                The state vector.

        Returns:
            state:
                The pure state of shape `(dimension,)` or the density matrix of shape `(dimension, dimension)`.
        """
        return np.asarray(self.states.read(x, self.states.names[0]))

    def _expect(self, terms: Iterable[OperatorKernel], t: float, x: Array, p: Array) -> float | complex:
        """Return the expectation value of an operator at a state of an operator problem.

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
            total += term.run(RunStateSpec(t, np.zeros(0), p)).values_ * complex(value)
        if abs(total.imag) <= REAL_TOLERANCE * (1 + abs(total.real)):
            return float(total.real)
        return total

    def _generator_at(self, t: float, p: Array) -> sp.csr_array:
        """Return the generator matrix of an operator problem.

        It is taken at a time and a parameter vector.

        Args:
            t:
                The time.

            p:
                The parameter vector.

        Returns:
            matrix:
                The sparse generator.
        """
        return OperatorProblemSpec._matrix_at(self.generator, t, p, math.prod(self.dimensions))
