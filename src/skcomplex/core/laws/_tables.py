"""Define the rule tables of discrete sites: next values by neighbourhood pattern, by sum, or by value and sum."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Table:
    """Rule given by the next value of every neighbourhood pattern of integer states `0` to `states - 1`.

    Args:
        mapping:
            A mapping from pattern, a tuple of values in neighbourhood order, to the next value.

        default:
            The next value of patterns the mapping leaves out.

        states:
            The number of state values, or `None` for one more than the largest value in the patterns.

    Examples:
        >>> from skcomplex.core.laws import Table
        >>> Table({(1, 1): 1}).n_states
        2
    """

    mapping: Mapping[tuple[int, ...], float]
    default: float = 0.0
    states: int | None = None

    @property
    def n_states(self) -> int:
        """Number of state values, the given `states` or one more than the largest value in the patterns, at least 2."""
        if self.states is not None:
            return self.states
        largest = max((value for pattern in self.mapping for value in pattern), default=0)
        return max(int(largest) + 1, 2)


@dataclass(frozen=True)
class Totalistic:
    """Rule given by the next value of every sum of the neighbourhood values.

    Args:
        rule:
            A mapping from sum to next value, or a function of the sums that returns the next values.

        default:
            The next value of sums a mapping leaves out.

        states:
            The number of state values `0` to `states - 1`, which bounds the sums.

    Examples:
        >>> from skcomplex.core.laws import Totalistic
        >>> Totalistic(lambda total: total > 4).states
        2
    """

    rule: Mapping[int, float] | Callable[[Any], Any]
    default: float = 0.0
    states: int = 2


@dataclass(frozen=True)
class OuterTotalistic:
    """Rule given by the next value of every pair of the cell's value and the sum of its neighbours' values.

    Args:
        rule:
            A mapping from `(value, sum)` to next value, or a function of the values and the sums.

        default:
            The next value of pairs a mapping leaves out.

        states:
            The number of state values `0` to `states - 1`, which bounds the sums.

    Examples:
        >>> from skcomplex.core.laws import OuterTotalistic
        >>> parity = OuterTotalistic(lambda center, total: (center + total) % 2)
        >>> parity.states
        2
    """

    rule: Mapping[tuple[int, int], float] | Callable[[Any, Any], Any]
    default: float = 0.0
    states: int = 2
