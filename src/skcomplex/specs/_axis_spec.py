"""Define the spec of a class axis, a uniform grid over an interval or a periodic range."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class AxisSpec(BaseSpec):
    """Spec of a class axis of a phase space, a uniform grid of cells over a range of values.

    Args:
        name:
            The name of the axis, which laws read its coordinate by.

        size:
            The number of cells.

        low:
            The lower end of the range.

        high:
            The upper end of the range.

        periodic:
            Whether the two ends of the range are joined, as for an angle, rather than edges without flux.

    Examples:
        >>> from skcomplex.specs import AxisSpec
        >>> AxisSpec('v', 128, -6.0, 6.0).name
        'v'
    """

    name: str
    size: int
    low: float
    high: float
    periodic: bool = False

    def _check(self) -> None:
        """Raise an error for a name that is no identifier, a size below one, or an empty range.

        Raises:
            ValidationError:
                The name is no identifier, the size is not a positive integer, or the range is empty.
        """
        if not isinstance(self.name, str) or not self.name.isidentifier():
            error_msg = f'The name of a class axis is an identifier, but it is {self.name!r}.'
            raise ValidationError(error_msg)
        if int(self.size) != self.size or self.size < 1:
            error_msg = f'The axis {self.name!r} has a positive number of cells, but its size is {self.size!r}.'
            raise ValidationError(error_msg)
        if not self.low < self.high:
            error_msg = f'The axis {self.name!r} runs from {self.low} to {self.high}. Give a low below the high.'
            raise ValidationError(error_msg)
