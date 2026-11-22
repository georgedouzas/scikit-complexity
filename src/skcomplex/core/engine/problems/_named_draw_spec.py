"""Define the named draw spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

import numpy.typing as npt

from ...base import BaseSpec
from ...components.laws import DrawSpec
from ...types import Array


@dataclass(frozen=True)
class _NamedDrawSpec(BaseSpec):
    """A spec of a named draw of a stochastic map, as runs record and replay it.

    Args:
        label:
            The path of the component and the name of the draw, such as `walker.kick`.

        draw:
            The draw, with its kind and distribution.

        start:
            The index of its first value in the extended parameter vector.

        rows:
            The number of values, one per member.

        arguments:
            The function `arguments(t, x, q)` that returns the values of the arguments of the distribution, given
            the extended parameter vector `q`, or `None` when they read more than the states and parameters of the
            member, so runs cannot record the draw.
    """

    label: str
    draw: DrawSpec
    start: int
    rows: int
    arguments: Callable[[float, Array, Array], list[npt.ArrayLike]] | None
