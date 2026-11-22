"""Define the diffusive limit spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...base import BaseSpec
from ._expression_printer import ExpressionPrinter
from ._reference_spec import ReferenceSpec


@dataclass(frozen=True, eq=False)
class DiffusiveLimitSpec(BaseSpec):
    """A spec of the largest stable time step of explicit diffusion on a grid.

    An explicit step of diffusion blows up when it is longer than `1 / (stiffness * D)`, where `D` is the diffusion
    coefficient and the stiffness `sum(2 / h^2)` over the axes grows as the cells shrink. The spec computes that step at
    the current parameters, so a solver keeps below it. It limits the steps of heat in a plate, a chemical in a gel, and
    a population that spreads over a landscape.

    Read more in the user guide, under Solvers.

    Args:
        coefficient:
            The NumPy function of time and parameters that returns the diffusion coefficient.

        parameters:
            The references of the parameters the coefficient reads, in argument order.

        stiffness:
            The largest eigenvalue magnitude of the unit Laplacian, `sum(2 / h^2)` over the axes.

    See Also:
        LinearTermSpec: A term of a field rate such as diffusion.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.engine.problems._diffusive_limit_spec import DiffusiveLimitSpec
        >>> DiffusiveLimitSpec(lambda t, *parameters: [0.5], (), 4.0).step(np.zeros(0))
        0.5
    """

    coefficient: Callable[..., object]
    parameters: tuple[ReferenceSpec, ...]
    stiffness: float

    def step(self, p: npt.NDArray[np.float64]) -> float:
        """Return the largest stable step of explicit Euler for the diffusion.

        The step holds at the parameter values.

        Args:
            p:
                The parameter vector.

        Returns:
            step:
                The step `1 / (stiffness * max |D|)`, infinite without diffusion.
        """
        values = [reference.get_column_of(np.zeros(0), p) for reference in self.parameters]
        magnitude = float(np.max(np.abs(ExpressionPrinter._get_values_of(self.coefficient(0.0, *values))[0])))
        return math.inf if magnitude == 0 else 1 / (self.stiffness * magnitude)
