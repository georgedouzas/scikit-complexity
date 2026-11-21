"""Define the totalistic spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy.typing as npt

from ...base import BaseSpec
from ...types import IntArray


@dataclass(frozen=True)
class TotalisticSpec(BaseSpec):
    """A spec of a local rule given by the next value of every sum of neighbours.

    It gives the next value of a cell from the sum of the values of its neighbours alone, such as
    `TotalisticSpec(lambda total: total % 2)` with a neighbourhood that holds the cell, where a cell becomes 1 when an
    odd number of the cells around it, itself included, are 1. The rule is a mapping from the sum to the next value,
    or a function of the array of sums, and the values run from `0` to `states - 1`. It runs a majority vote of
    neighbours, the growth of a crystal, and the spread of a rumour.

    Read more in the user guide, under Laws.

    Args:
        rule:
            A mapping from sum to next value, or a function of the sums that returns the next values.

        default:
            The next value of sums a mapping leaves out.

        states:
            The number of values of a cell, `0` to `states - 1`, which bounds the sums.

    See Also:
        OuterTotalisticSpec: A local rule of the value of a cell and the sum of its neighbours.
        UpdateLaw: Takes the spec as the value of a local rule.

    Examples:
        >>> from skcomplex.core.components.laws import TotalisticSpec
        >>> TotalisticSpec(lambda total: total > 4).states
        2
    """

    rule: Mapping[int, float] | Callable[[IntArray], npt.ArrayLike]
    default: float = 0.0
    states: int = 2
