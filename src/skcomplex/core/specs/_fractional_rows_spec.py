"""Define the fractional rows spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..types import Array
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class FractionalRowsSpec(BaseSpec):
    """Noise rows of one fractional noise law.

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
