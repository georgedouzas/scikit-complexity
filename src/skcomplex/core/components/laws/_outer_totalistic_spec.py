"""Define the outer totalistic spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy.typing as npt

from ...base import BaseSpec
from ...types import IntArray


@dataclass(frozen=True)
class OuterTotalisticSpec(BaseSpec):
    """A spec of a local rule of the value of a cell and the sum of its neighbours.

    It gives the next value of a cell from its own value and the sum of the values of its neighbours, such as the Game
    of Life, `OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})` with `neighborhood=MOORE`, where a dead cell with
    three live neighbours comes alive and a live cell with two or three stays alive. The rule is a mapping from the
    pair of the value and the sum to the next value, or a function of the arrays of values and sums. It runs the Game
    of Life, the spread of a forest fire, and the waves of a beating heart.

    Read more in the user guide, under Laws.

    Args:
        rule:
            A mapping from the pair `(value, sum)` to the next value, or a function of the values and the sums.

        default:
            The next value of pairs a mapping leaves out.

        states:
            The number of values of a cell, `0` to `states - 1`, which bounds the sums.

    See Also:
        TotalisticSpec: A local rule of the sum of the neighbours alone.
        UpdateLaw: Takes the spec as the value of a local rule.

    Examples:
        >>> from skcomplex.core.components.laws import OuterTotalisticSpec
        >>> life = OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1})
        >>> life.rule[(0, 3)], life.default
        (1, 0.0)
    """

    rule: Mapping[tuple[int, int], float] | Callable[[IntArray, IntArray], npt.ArrayLike]
    default: float = 0.0
    states: int = 2
