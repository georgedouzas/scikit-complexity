"""Define the spec of an extra axis of a phase space, a range cut into equal cells."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...errors import ValidationError


@dataclass(frozen=True)
class AxisSpec(BaseSpec):
    """A spec of an extra axis of a phase space, a range cut into equal cells.

    It cuts the range from `low` to `high` into `size` cells of width `(high - low) / size`, so
    `AxisSpec('v', 128, -6.0, 6.0)` gives a phase space a velocity axis of 128 cells. A periodic axis joins its two
    ends, as an angle does. It holds the velocity of gas molecules, the swimming direction of bacteria, or the age of
    the cells of a tissue.

    Read more in the user guide, under Spaces.

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

    See Also:
        PhaseSpace: Combines a lattice of positions with these axes.

    Examples:
        >>> from skcomplex.core.components.spaces import AxisSpec, LatticeSpace, PhaseSpace
        >>> space = PhaseSpace(LatticeSpace(4), [AxisSpec('v', 3, -1.5, 1.5)]).prepare()
        >>> space.coordinates_['v'].tolist()
        [-1.0, 0.0, 1.0]
    """

    name: str
    size: int
    low: float
    high: float
    periodic: bool = False

    def _check(self) -> None:
        """Raise an error for an invalid name, size, or range.

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
