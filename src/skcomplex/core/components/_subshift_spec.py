"""Define the subshift spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ..base import BaseSpec
from ..errors import ValidationError


@dataclass(frozen=True)
class SubshiftSpec(BaseSpec):
    """A spec of the values of cells that no forbidden pattern may appear in.

    The cells take the values `0` to `states - 1`, and a pattern is a tuple of values over the cells of a neighbourhood,
    such as `(1, 1)` for two live cells side by side. A configuration is allowed when no forbidden pattern appears at
    any site, and patterns that reach past the edge of a space are not checked. A run checks that every cell holds a
    value of `alphabet`, and `automaton` decides whether a whole configuration is allowed. It describes the golden mean
    shift of symbolic dynamics, hard core gases, and codes that forbid long runs of one symbol.

    Read more in the user guide, under Elements.

    Args:
        neighborhood:
            The neighbourhood of offsets on a lattice, or of elements on a group space, that patterns cover.

        forbidden:
            The forbidden patterns.

        states:
            The number of values of a cell, `0` to `states - 1`.

    See Also:
        LatticeNeighborhood: The offsets a pattern covers.

    Examples:
        >>> from skcomplex.core.components import SubshiftSpec
        >>> from skcomplex.core.components.spaces import LatticeNeighborhood
        >>> golden = SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),))
        >>> golden.forbidden, golden.states
        (((1, 1),), 2)
    """

    neighborhood: object
    forbidden: Sequence[tuple[int, ...]]
    states: int = 2

    def _check(self) -> None:
        """Raise an error for forbidden patterns of different lengths or with bad values.

        A bad value lies outside the states.

        Raises:
            ValidationError:
                The forbidden patterns differ in length, or hold a value outside `0` to `states - 1`.
        """
        if len({len(pattern) for pattern in self.forbidden}) > 1:
            error_msg = f'The forbidden patterns {self.forbidden!r} differ in length. Give one length, the neighbours.'
            raise ValidationError(error_msg)
        if any(not 0 <= value < self.states for pattern in self.forbidden for value in pattern):
            error_msg = f'The forbidden patterns {self.forbidden!r} hold values outside 0 to {self.states - 1}.'
            raise ValidationError(error_msg)

    @property
    def alphabet(self) -> FiniteEnumeratedSet:
        """Sage set of the values of a cell."""
        return FiniteEnumeratedSet(list(range(self.states)))
