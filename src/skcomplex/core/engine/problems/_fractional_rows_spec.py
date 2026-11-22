"""Define the fractional rows spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...base import BaseSpec
from ...types import Array


@dataclass(frozen=True)
class _FractionalRowsSpec(BaseSpec):
    """A spec of the noise rows of one noise law with a Hurst index.

    The law is driven by fractional Brownian motion of a Hurst index.

    Args:
        law:
            The description of the law, such as `fractional noise law of walkers.x`.

        positions:
            The positions of the law's rows among the noise rows of the problem.

        hurst:
            The function `H(p)` that returns the Hurst index of every row from the parameter vector.
    """

    law: str
    positions: npt.NDArray[np.intp]
    hurst: Callable[[Array], Array]
