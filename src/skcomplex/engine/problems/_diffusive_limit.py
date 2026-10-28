"""Stability limit of explicit steps for the diffusion of a finite-volume field, at the current parameters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ._base import BaseValue
from ._reference import Reference


@dataclass(frozen=True, eq=False)
class DiffusiveLimit(BaseValue):
    """Stability limit of explicit steps for the diffusion of a finite-volume field, at the current parameters.

    Args:
        coefficient:
            The NumPy function of time and parameters that returns the diffusion coefficient.

        parameters:
            The references of the parameters the coefficient reads, in argument order.

        stiffness:
            The largest eigenvalue magnitude of the unit Laplacian, `sum(2 / h^2)` over the axes.
    """

    coefficient: Callable[..., list[Any]]
    parameters: tuple[Reference, ...]
    stiffness: float

    def step(self, p: npt.NDArray[np.float64]) -> float:
        """Return the largest stable step of explicit Euler for the diffusion at parameter values.

        Args:
            p:
                The parameter vector.

        Returns:
            step:
                The step `1 / (stiffness * max |D|)`, infinite without diffusion.
        """
        values = [reference.column_of(np.zeros(0), p) for reference in self.parameters]
        magnitude = float(np.max(np.abs(self.coefficient(0.0, *values)[0])))
        return math.inf if magnitude == 0 else 1 / (self.stiffness * magnitude)
