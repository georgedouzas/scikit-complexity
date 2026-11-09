"""Sparse matrix on the whole Hilbert space with a symbolic coefficient.

It holds the lowered function of the coefficient.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Iterable

import numpy as np
import scipy.sparse as sp
from sage.all__sagemath_symbolics import Expression

from ...specs import RunStateSpec
from ...types import Array
from .._utils import symbol
from ._base import BaseKernel


class OperatorKernel(BaseKernel):
    """Sparse matrix on the whole Hilbert space with a symbolic coefficient.

    It holds the lowered function of the coefficient.

    Args:
        matrix:
            The sparse matrix.

        coefficient:
            The symbolic coefficient, 1 for a constant matrix.

        function:
            The function `c(t, p)` of the coefficient, or `None` for a constant matrix.

    Attributes:
        values_:
            The coefficient at the time and the parameters of the state of the last run.
    """

    values_: complex

    def __init__(
        self,
        matrix: sp.csr_array,
        coefficient: Expression,
        function: Callable[[float, Array], complex] | None = None,
    ) -> None:
        self.matrix = matrix
        self.coefficient = coefficient
        self.function = function

    @property
    def time_dependent(self) -> bool:
        """Whether the coefficient depends on time."""
        return bool(self.coefficient.has(symbol('t')))

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the coefficient at the time and the parameters of the state.

        A constant matrix has coefficient 1.

        Args:
            inputs:
                The state.
        """
        state = self._get_state_of(inputs)
        self.values_ = 1 + 0j if self.function is None else self.function(state.t, state.p)

    @staticmethod
    def _get_coefficient_of(value: object) -> complex:
        """Return the value a lowered coefficient function computed.

        The value is returned as a complex coefficient.

        Args:
            value:
                The computed value.

        Returns:
            coefficient:
                The complex coefficient.

        Raises:
            TypeError:
                The value is not a number.
        """
        if not isinstance(value, int | float | complex | np.number):
            error_msg = f'A coefficient function computes a number, not a {type(value).__name__}.'
            raise TypeError(error_msg)
        return complex(value)

    @staticmethod
    def _sum_matrices(matrices: Iterable[sp.csr_array], size: int) -> sp.csr_array:
        """Return the sum of sparse matrices, the zero matrix without any.

        Args:
            matrices:
                The matrices on the whole space.

            size:
                The dimension of the whole space.

        Returns:
            total:
                The sum.
        """
        total = sp.csr_array((size, size), dtype=complex)
        for matrix in matrices:
            total = total + matrix
        return sp.csr_array(total)
