"""Define the table spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from ._base_spec import BaseSpec


@dataclass(frozen=True)
class TableSpec(BaseSpec):
    """A spec of a rule given by the next value of every neighbourhood pattern.

    The states are `0` to `states - 1`.

    Args:
        mapping:
            A mapping from pattern, a tuple of values in neighbourhood order, to the next value.

        default:
            The next value of patterns the mapping leaves out.

        states:
            The number of state values, or `None` for one more than the largest value in the patterns.

    Examples:
        >>> from skcomplex.core.specs import TableSpec
        >>> TableSpec({(1, 1): 1}).n_states
        2
    """

    mapping: Mapping[tuple[int, ...], float]
    default: float = 0.0
    states: int | None = None

    @property
    def n_states(self) -> int:
        """Number of state values.

        It is the given `states` or one more than the largest value in the patterns, at least 2.
        """
        if self.states is not None:
            return self.states
        largest = max((value for pattern in self.mapping for value in pattern), default=0)
        return max(int(largest) + 1, 2)
