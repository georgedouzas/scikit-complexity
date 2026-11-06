"""Define the people spec of Metaphoria."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from ....core.specs import BaseSpec
from ._settler_view import SettlerView


@dataclass(frozen=True)
class _PeopleSpec(BaseSpec):
    """Living people by id at one time, which the rules build once per step and keep in the storage of the system.

    Args:
        time:
            The time the people were read at.

        people:
            The living people by id.
    """

    time: float
    people: Mapping[int, SettlerView]
