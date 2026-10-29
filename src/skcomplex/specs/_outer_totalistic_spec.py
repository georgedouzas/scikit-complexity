"""Define the outer totalistic spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy.typing as npt

from ..types import IntArray
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class OuterTotalisticSpec(BaseSpec):
    """Rule given by the next value of every pair of the cell's value and the sum of its neighbours' values.

    Args:
        rule:
            A mapping from `(value, sum)` to next value, or a function of the values and the sums.

        default:
            The next value of pairs a mapping leaves out.

        states:
            The number of state values `0` to `states - 1`, which bounds the sums.

    Examples:
        >>> from skcomplex.specs import OuterTotalisticSpec
        >>> parity = OuterTotalisticSpec(lambda center, total: (center + total) % 2)
        >>> parity.states
        2
    """

    rule: Mapping[tuple[int, int], float] | Callable[[IntArray, IntArray], npt.ArrayLike]
    default: float = 0.0
    states: int = 2
