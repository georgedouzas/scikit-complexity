"""Sparse matrix on the whole Hilbert space with a symbolic coefficient and its lowered function."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import scipy.sparse as sp
from sage.all__sagemath_symbolics import Expression

from ....core.types import Vector
from ...problems import (
    symbol,
)


@dataclass(frozen=True)
class QuantumTerm:
    """Sparse matrix on the whole Hilbert space with a symbolic coefficient and its lowered function.

    Args:
        matrix:
            The sparse matrix.

        coefficient:
            The symbolic coefficient, 1 for a constant matrix.

        function:
            The function `c(t, p)` of the coefficient, or `None` for a constant matrix.
    """

    matrix: sp.csr_array
    coefficient: Expression
    function: Callable[[float, Vector], complex] | None = None

    @property
    def time_dependent(self) -> bool:
        """Whether the coefficient depends on time."""
        return bool(self.coefficient.has(symbol('t')))

    def value(self, t: float, p: Vector) -> complex:
        """Return the coefficient at a time and a parameter vector.

        Args:
            t:
                The time.

            p:
                The parameter vector.

        Returns:
            value:
                The coefficient.
        """
        return 1.0 if self.function is None else self.function(t, p)
