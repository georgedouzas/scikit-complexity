"""Define the delayed state spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression

from ...specs._base_spec import BaseSpec
from ...specs._entry_spec import EntrySpec
from ._kernel_spec import KernelSpec


@dataclass(frozen=True)
class _DelayedStateSpec(BaseSpec):
    """State a law reads with a delay.

    The law reads it through the hidden parameter that solvers fill from its history.

    Args:
        entry:
            The entry of the hidden parameter.

        states:
            The indices of the delayed state in the state vector.

        tau:
            The delay.

        minimum:
            The smallest value of the delay.

        spec:
            The kernel spec of the law that reads the delayed state.

        symbol:
            The symbol of the hidden parameter in the law.

        state:
            The symbol of the delayed state.
    """

    entry: EntrySpec
    states: npt.NDArray[np.intp]
    tau: Expression
    minimum: float
    spec: KernelSpec
    symbol: Expression
    state: Expression
