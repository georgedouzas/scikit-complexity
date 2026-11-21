"""Define the period a value of an element wraps around."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from dataclasses import dataclass

from ...base import BaseSpec
from ...errors import ValidationError


@dataclass(frozen=True)
class Period(BaseSpec):
    """A spec of a value that wraps around a period, such as `2 * pi` for an angle.

    It is written in `Annotated` beside the allowed values, such as `theta: Annotated[float, Period(2 * pi)] = 0.0`. The
    value is brought back within one period after every step, so `0` and `2 * pi` are the same angle. The analyses read a
    periodic value as a phase, or as a heading when its element moves in a space. It holds the angle of a pendulum, the
    phase of a firefly, and the heading of a bird.

    Read more in the user guide, under Elements.

    Args:
        length:
            The length of the period, a positive number.

    See Also:
        Quantity: Marks a value whose total only flows change.
        Tensor: Marks a value that holds a square matrix.

    Examples:
        >>> import math
        >>> from skcomplex.core.components.elements import Period
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
