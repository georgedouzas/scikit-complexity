"""Define the base record spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class BaseRecordSpec(BaseSpec):
    """Base of the records: something that happened at one time of a run, such as an event or a relaxation.

    Every record has the readers `label`, the name of what happened, and `time`, the time it happened at.
    """
