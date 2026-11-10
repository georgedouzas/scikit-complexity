"""Define the cascade record spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_record_spec import BaseRecordSpec


@dataclass(frozen=True)
class CascadeRecordSpec(BaseRecordSpec):
    """A spec of the record of a cascade of a lattice within one step.

    Args:
        label:
            The label of the cascade.

        time:
            The time of the step.

        size:
            The number of site updates.

        duration:
            The number of iterations of the cascade.

        area:
            The number of distinct sites that changed.
    """

    label: str
    time: float
    size: int
    duration: int
    area: int
