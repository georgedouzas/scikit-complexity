"""Define the named draw spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from dataclasses import dataclass

from ..types import Array
from ._base_spec import BaseSpec
from ._draw_spec import DrawSpec


@dataclass(frozen=True)
class NamedDrawSpec(BaseSpec):
    """Draw of a component that a stochastic map takes, as runs record and replay it.

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
    arguments: Callable[[float, Array, Array], list[Array]] | None
