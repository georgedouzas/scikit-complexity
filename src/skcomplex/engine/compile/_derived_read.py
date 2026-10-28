"""Value of a global operator, a spectral multiplier or a finite element matrix, applied to an expression."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ..problems import numpy_function, symbol


@dataclass(eq=False)
class DerivedRead:
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

    operand: Expression
    symbols: list[Expression]
    apply: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    multiplier: npt.NDArray[np.complex128] | None = None
    dimensions: tuple[int, ...] | None = None
    matrix: Any = None
    _function: Callable[..., Any] | None = field(default=None, repr=False)

    def evaluate(self, t: float, values: list[Any], size: int) -> npt.NDArray[np.float64]:
        """Return the operator applied to the operand evaluated from the values of its symbols.

        Args:
            t:
                The time.

            values:
                The values of the operand's symbols.

            size:
                The number of sites.

        Returns:
            values:
                The operator's value at every site.
        """
        if self._function is None:
            self._function = numpy_function(symbol('t'), self.symbols, [self.operand])
        operand = np.broadcast_to(np.asarray(self._function(t, *values)[0], dtype=float), (size,))
        return self.apply(operand)
