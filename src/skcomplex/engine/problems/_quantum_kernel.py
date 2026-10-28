"""Sparse matrix on the whole Hilbert space with a symbolic coefficient and its lowered function."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable

import scipy.sparse as sp
from sage.all__sagemath_symbolics import Expression

from ...types import Array
from .._utils import symbol
from ._base import BaseKernel
from ._state import State


class QuantumKernel(BaseKernel):
    """Sparse matrix on the whole Hilbert space with a symbolic coefficient and its lowered function.

    Args:
        matrix:
            The sparse matrix.

        coefficient:
            The symbolic coefficient, 1 for a constant matrix.

        function:
            The function `c(t, p)` of the coefficient, or `None` for a constant matrix.
    """

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

    def _run(self, inputs: State) -> complex:
        """Return the coefficient at the time and the parameters of the state."""
        return 1.0 if self.function is None else self.function(inputs.t, inputs.p)
