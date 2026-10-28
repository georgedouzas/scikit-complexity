"""Event that happened during a run."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

from ...types import Array
from ._base import BaseRecord


@dataclass(frozen=True)
class EventRecord(BaseRecord):
    """Event that happened during a run.

    Args:
        label:
            The name of the event.

        time:
            The time of the crossing.

        before:
            The state vector at the crossing, before the reset.

        after:
            The state vector after the reset.
    """

    label: str
    time: float
    before: Array
    after: Array
