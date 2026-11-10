"""Define the totalistic spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy.typing as npt

from ..types import IntArray
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class TotalisticSpec(BaseSpec):
    """A spec of a rule given by the next value of every neighbourhood sum.

    Args:
        rule:
            A mapping from sum to next value, or a function of the sums that returns the next values.

        default:
            The next value of sums a mapping leaves out.

        states:
            The number of state values `0` to `states - 1`, which bounds the sums.

    Examples:
        >>> from skcomplex.core.specs import TotalisticSpec
        >>> TotalisticSpec(lambda total: total > 4).states
        2
    """

    rule: Mapping[int, float] | Callable[[IntArray], npt.ArrayLike]
    default: float = 0.0
    states: int = 2
