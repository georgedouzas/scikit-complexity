"""Define the clamp spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class ClampSpec(BaseSpec):
    """Clamp of the change of a vector, by the largest angle it turns and the bounds of its norm.

    Every bound is a number, the name of a parameter of the agent, or `None` for no bound. The vector turns from its
    current direction toward its new one by at most `max_angle`, keeping the new norm, and its norm is then clipped
    between `min_norm` and `max_norm`. A zero vector stays zero.

    Args:
        max_angle:
            The largest angle in radians the vector turns, or `None`.

        min_norm:
            The smallest norm, or `None`.

        max_norm:
            The largest norm, or `None`.

    Raises:
        ValidationError:
            A bound is negative, raised by `prepare`.

    Examples:
        >>> from skcomplex.specs import ClampSpec
        >>> ClampSpec(min_norm=0.5, max_norm='speed').max_norm
        'speed'
    """

    max_angle: float | str | None = None
    min_norm: float | str | None = None
    max_norm: float | str | None = None

    def _check(self) -> None:
        """Raise an error for a negative bound.

        Raises:
            ValidationError:
                A bound is negative.
        """
        for name, value in (('max_angle', self.max_angle), ('min_norm', self.min_norm), ('max_norm', self.max_norm)):
            if value is not None and not isinstance(value, str) and not value >= 0:
                error_msg = f'The bound {name} of a clamp is {value!r}. Give a nonnegative number, a name, or None.'
                raise ValidationError(error_msg)
