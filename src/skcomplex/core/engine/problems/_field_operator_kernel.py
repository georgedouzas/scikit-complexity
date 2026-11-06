"""Value of a global operator, a spectral multiplier or a finite element matrix, applied to an expression."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import Expression

from ...specs import RunStateSpec
from .._utils import symbol
from ._base import BaseKernel
from ._expression_printer import ExpressionPrinter


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

        displacement:
            The displacement along every axis of a read at shifted positions, expressions of the operand's symbols
            that `apply` takes after the operand's values, or `None`.

    Attributes:
        values_:
            The operator applied to the operand values at every site of the last run.
    """

    values_: npt.NDArray[np.float64]

    def __init__(
        self,
        operand: Expression,
        symbols: list[Expression],
        apply: Callable[..., npt.NDArray[np.float64]],
        multiplier: npt.NDArray[np.complex128] | None = None,
        dimensions: tuple[int, ...] | None = None,
        *,
        matrix: scipy.sparse.sparray | None = None,
        displacement: list[Expression] | None = None,
    ) -> None:
        self.operand = operand
        self.symbols = symbols
        self.apply = apply
        self.multiplier = multiplier
        self.dimensions = dimensions
        self.matrix = matrix
        self.displacement = displacement

    @functools.cached_property
    def _function(self) -> Callable[..., object]:
        """NumPy function of the operand and the displacement, compiled once."""
        printer = ExpressionPrinter(time=symbol('t'), compiled=True).prepare(self.symbols)
        return ExpressionPrinter._function_of(printer.run([self.operand, *(self.displacement or [])]).code_)

    def operand_values(self, t: float, values: Sequence[object], size: int) -> npt.NDArray[np.float64]:
        """Return the operand at every site, evaluated from the values of its symbols, stacked over the displacement.

        Args:
            t:
                The time.

            values:
                The values of the operand's symbols.

            size:
                The number of sites.

        Returns:
            operand:
                The operand's value at every site, or one row of the operand and one per axis of the displacement.
        """
        computed = [
            np.broadcast_to(np.asarray(item, dtype=float), (size,))
            for item in ExpressionPrinter._values_of(self._function(t, *values))
        ]
        return computed[0] if self.displacement is None else np.stack(computed)

    def _run(self, inputs: RunStateSpec | None) -> None:
        """Set the operator applied to the operand values at every site, the `x` of `inputs`."""
        state = self._state_of(inputs)
        self.values_ = self.apply(state.x) if self.displacement is None else self.apply(state.x[0], state.x[1:])
