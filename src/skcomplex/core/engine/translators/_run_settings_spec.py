"""Define the run settings spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...base import BaseSpec
from ...errors import ValidationError


@dataclass(frozen=True)
class RunSettingsSpec(BaseSpec):
    """A spec of the run settings of a model file: its start, length, and steps.

    It holds the time a run of the file starts at, how long it runs, its time step, and how often it saves values, such
    as a run from the year 1900 for 100 years in steps of a quarter. A translator reads it from the control section of
    the file and writes it as the constants of the system module. It sets the run of a population model, a supply chain,
    and a climate model alike.

    Read more in the user guide, under Systems.

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
            The length of the run, the time step, or the saving interval is not positive, raised when its owner checks
            it.

    See Also:
        SystemDynamicsTranslator: Reads the settings from a model file.

    Examples:
        >>> from skcomplex.core.engine.translators import RunSettingsSpec
        >>> RunSettingsSpec(start=1900.0, until=100.0, dt=0.25, save_every=1.0).until
        100.0
    """

    start: float
    until: float
    dt: float
    save_every: float

    def _check(self) -> None:
        """Raise an error for a run length, time step, or saving interval not positive.

        Raises:
            ValidationError:
                The length of the run, the time step, or the saving interval is not positive.
        """
        for name in ('until', 'dt', 'save_every'):
            value = getattr(self, name)
            if not value > 0:
                error_msg = f'The run setting {name} is {value!r}. Give a positive number.'
                raise ValidationError(error_msg)
