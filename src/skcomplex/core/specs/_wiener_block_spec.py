"""Define the wiener block spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class WienerBlockSpec(BaseSpec):
    """Correlated Wiener processes of one `WienerProcessSpec`, a contiguous block of the processes of a problem.

    Args:
        start:
            The index of the first process of the block.

        size:
            The number of processes of the block.

        factor:
            The factor `L` with `L L^T = C` of the covariance `C` of the block.
    """

    start: int
    size: int
    factor: npt.NDArray[np.float64]
