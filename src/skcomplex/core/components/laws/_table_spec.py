"""Define the table spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from ...base import BaseSpec


@dataclass(frozen=True)
class TableSpec(BaseSpec):
    """A spec of a local rule given by the next value of every pattern of neighbours.

    It lists the next value of a cell for every pattern of the values of its neighbours, in the order of the
    neighbourhood, such as rule 30, `TableSpec({(1, 0, 0): 1, (0, 1, 1): 1, (0, 1, 0): 1, (0, 0, 1): 1})`, where a cell
    reads its left neighbour, itself, and its right neighbour. Patterns the mapping leaves out take `default`, and the
    values run from `0` to `states - 1`. It runs the elementary automata, draws the patterns of seashells, and makes
    the random numbers of rule 30.

    Read more in the user guide, under Laws.

    Args:
        mapping:
            A mapping from a pattern, a tuple of values in the order of the neighbourhood, to the next value.

        default:
            The next value of patterns the mapping leaves out.

        states:
            The number of values of a cell, or `None` for one more than the largest value in the patterns.

    See Also:
        TotalisticSpec: A local rule of the sum of the neighbours.
        UpdateLaw: Takes the spec as the value of a local rule.

    Examples:
        >>> from skcomplex.core.components.laws import TableSpec
        >>> rule30 = TableSpec({(1, 0, 0): 1, (0, 1, 1): 1, (0, 1, 0): 1, (0, 0, 1): 1})
        >>> rule30.n_states, rule30.default
        (2, 0.0)
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
