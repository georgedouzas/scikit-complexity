"""Renewals of one renewal law that fired in one pass, each at the time of its member."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...core.types import Vector


@dataclass(frozen=True)
class RenewalRecord:
    """Renewals of one renewal law that fired in one pass, each at the time of its member.

    Args:
        label:
            The name of the renewal law.

        times:
            The time of every renewal.

        members:
            The member of every renewal.

        before:
            The states of every renewing member before its reset, one row per renewal and one column per state.

        after:
            The states of every renewing member after its reset.

        states:
            The dotted state name, with the component of a vector state, of every column.
    """

    label: str
    times: Vector
    members: npt.NDArray[np.intp]
    before: Vector
    after: Vector
    states: tuple[str, ...]

    @property
    def time(self) -> float:
        """Time of the last renewal of the pass."""
        return float(np.max(self.times))
