"""Define the run settings spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class RunSettingsSpec(BaseSpec):
    """Run settings of a model file: where its time starts, how long it runs, its step, and its saving interval.

    Args:
        start:
            The time of the file at the start of a run.

        until:
            The length of a run, the end time of the file minus `start`.

        dt:
            The time step.

        save_every:
            The time between saved values.

    Raises:
        ValidationError:
            The length of the run, the time step, or the saving interval is not positive, raised by `check`.

    Examples:
        >>> from skcomplex.specs import RunSettingsSpec
        >>> RunSettingsSpec(start=1900.0, until=100.0, dt=0.25, save_every=1.0).check().until
        100.0
    """

    start: float
    until: float
    dt: float
    save_every: float

    def _check(self) -> None:
        """Raise an error for a run length, a time step, or a saving interval that is not positive.

        Raises:
            ValidationError:
                The length of the run, the time step, or the saving interval is not positive.
        """
        for name in ('until', 'dt', 'save_every'):
            value = getattr(self, name)
            if not value > 0:
                error_msg = f'The run setting {name} is {value!r}. Give a positive number.'
                raise ValidationError(error_msg)
