"""Value of a global operator, a spectral multiplier or a finite element matrix, applied to an expression."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import Expression

from ._base import BaseKernel
from ._printing import numpy_function
from ._state import State
from ._symbols import symbol


class FieldOperatorKernel(BaseKernel):
    """Value of a global operator, a spectral multiplier or a finite element matrix, applied to an expression.

    Args:
        operand:
            The expression of member symbols the operator acts on.

        symbols:
            The symbols of the operand in the order its function takes them.

        apply:
            The operator on the operand's values at every site.

        multiplier:
            The Fourier multiplier of a spectral operator, or `None`.

        dimensions:
            The lattice shape of a spectral operator, or `None`.

        matrix:
            The sparse matrix of an operator that acts on the sites as one, such as the finite element Laplacian, or
            `None` for spectral operators.
    """

    def __init__(  # noqa: PLR0917
        self,
        operand: Expression,
        symbols: list[Expression],
        apply: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        multiplier: npt.NDArray[np.complex128] | None = None,
        dimensions: tuple[int, ...] | None = None,
        matrix: scipy.sparse.sparray | None = None,
    ) -> None:
        self.operand = operand
        self.symbols = symbols
        self.apply = apply
        self.multiplier = multiplier
        self.dimensions = dimensions
        self.matrix = matrix

    @functools.cached_property
    def _function(self) -> Callable[..., Any]:
        """NumPy function of the operand, compiled once."""
        return numpy_function(symbol('t'), self.symbols, [self.operand])

    def operand_values(self, t: float, values: list[Any], size: int) -> npt.NDArray[np.float64]:
        """Return the operand at every site, evaluated from the values of its symbols.

        Args:
            t:
                The time.

            values:
                The values of the operand's symbols.

            size:
                The number of sites.

        Returns:
            operand:
                The operand's value at every site.
        """
        return np.broadcast_to(np.asarray(self._function(t, *values)[0], dtype=float), (size,))

    def _run(self, inputs: State) -> npt.NDArray[np.float64]:
        """Return the operator applied to the operand values at every site, the `x` of `inputs`."""
        return self.apply(inputs.x)
