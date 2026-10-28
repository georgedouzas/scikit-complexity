"""Delayed state read by laws through a hidden parameter that solvers fill from the run's history."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...types import Array
from ._base import BaseValue


@dataclass(frozen=True)
class _Delay(BaseValue):
    """Delayed state read by laws through a hidden parameter that solvers fill from the run's history."""

    parameters: npt.NDArray[np.intp]
    states: npt.NDArray[np.intp]
    tau: Callable[[float, Array, Array], float]
    minimum: float
    symbol: Expression
    state: Expression
    delay: Expression
