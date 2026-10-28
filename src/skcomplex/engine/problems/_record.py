"""Record of something that happened at one time of a run, such as an event or a relaxation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Protocol


class Record(Protocol):
    """Record of something that happened at one time of a run, such as an event or a relaxation."""

    @property
    def label(self) -> str:
        """Name of what happened."""

    @property
    def time(self) -> float:
        """Time it happened at."""
