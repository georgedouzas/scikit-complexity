"""Define the period a value of an element wraps around."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from dataclasses import dataclass

from ...base import BaseSpec
from ...errors import ValidationError


@dataclass(frozen=True)
class Period(BaseSpec):
    """A period that a value of an element wraps around, such as `2 * pi` for an angle.

    It is written in `Annotated` beside the allowed values, such as `theta: Annotated[float, Period(2 * pi)] = 0.0`.
    The value is brought back within one period after every step, so the angle of a pendulum, the phase of a clock, or
    the heading of a bird stays on its circle. The analyses read a periodic value as a phase, or as a heading when its
    element moves in a space.

    Unlike the allowed values in `Annotated`, which reject a value outside them, it brings a value back onto its
    circle. Choose it when a value is an angle, so `0` and `2 * pi` are the same value.

    It appears in many fields. The angle of a pendulum, the phase of a firefly, and the heading of a bird each wrap
    around.

    Args:
        length:
            The length of the period, a positive number.

    Examples:
        >>> import math
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import Element, Period
        >>> class Clock(Element):
        ...     def __init__(self, label, theta: Annotated[float, Period(2 * math.pi)] = 0.0):
        ...         super().__init__(label)
        ...         self.theta = theta
        >>> round(Period(2 * math.pi).length, 4)
        6.2832
    """

    length: float

    def _check(self) -> None:
        """Raise an error for a length that is not a positive finite number.

        A number of Python, NumPy, or Sage is accepted, such as `2 * pi` with the `pi` of Sage.

        Raises:
            ValidationError:
                The length is not a positive finite number.
        """
        try:
            length = float(self.length)
        except (TypeError, ValueError):
            length = math.nan
        positive = math.isfinite(length) and length > 0
        if not positive or isinstance(self.length, bool | str):
            error_msg = f'A period has a positive length, such as Period(2 * pi), but it is {self.length!r}.'
            raise ValidationError(error_msg)
