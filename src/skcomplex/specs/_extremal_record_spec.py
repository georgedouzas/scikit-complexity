"""Define the extremal record spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_record_spec import BaseRecordSpec


@dataclass(frozen=True, slots=True)
class ExtremalRecordSpec(BaseRecordSpec):
    """Application of an extremal law, such as the replacement of the least fit species.

    Args:
        label:
            The label of the extremal law.

        time:
            The time of the step.

        site:
            The member index of the extremal site, in row-major order on a lattice or node order on a network.

        value:
            The extremal value of the key state before the update.
    """

    label: str
    time: float
    site: int
    value: float
