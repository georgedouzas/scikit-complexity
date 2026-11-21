"""Define the renewal record spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...types import Array
from ._base_record_spec import BaseRecordSpec


@dataclass(frozen=True)
class RenewalRecordSpec(BaseRecordSpec):
    """A spec of the renewals of one renewal law that fired in one pass.

    It holds the label of the law, the time and the member of each renewal, such as the walkers that jumped, and the
    states before and after. It records the jumps of walkers in a glass, the spikes of neurons, and the repairs of
    machines.

    Read more in the user guide, under Analyse.

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

    See Also:
        BaseRecordSpec: The base of every record.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ContinuousTimeRandomWalkSystem
        >>> trajectory = ContinuousTimeRandomWalkSystem(n_walkers=3).prepare().simulate(until=10, random_state=0)
        >>> type(trajectory.events[0]).__name__, trajectory.events[0].label
        ('RenewalRecordSpec', 'jumps')
    """

    label: str
    times: Array
    members: npt.NDArray[np.intp]
    before: Array
    after: Array
    states: tuple[str, ...]

    @property
    def time(self) -> float:
        """Time of the last renewal of the pass."""
        return float(np.max(self.times))
