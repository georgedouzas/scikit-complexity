"""Successor of a production of a word, as symbol codes and the function of its parameters."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...specs import BaseSpec
from ...types import Arrays


@dataclass(frozen=True)
class _Successor(BaseSpec):
    """Successor of a production, as symbol codes and the function of its parameters."""

    codes: npt.NDArray[np.int64]
    slots: tuple[tuple[int, int], ...]
    values: Callable[..., Arrays] | None
    variables: int
