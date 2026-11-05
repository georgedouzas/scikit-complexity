"""Define the subshift spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import dataclass

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ..errors import ValidationError
from ._base_spec import BaseSpec


@dataclass(frozen=True)
class SubshiftSpec(BaseSpec):
    """Domain of a state given by a subshift of finite type: the configurations without a forbidden pattern.

    A pattern is a tuple of values in the order of a neighbourhood. It appears at a site when the cells of the
    neighbourhood of the site hold its values, and patterns that reach outside a space are not checked, so a lattice
    with fixed edges holds the words of a window. A run checks that every cell holds a value of `alphabet`, and
    `AutomatonAnalysis` decides whether a whole configuration is admissible.

    Args:
        neighborhood:
            The neighbourhood of offsets on a lattice, or the memory set on a group space, that patterns cover.

        forbidden:
            The forbidden patterns.

        states:
            The number of values of a cell, `0` to `states - 1`.

    Examples:
        >>> from skcomplex.core.spaces import LatticeNeighborhood
        >>> from skcomplex.specs import StateSpec, SubshiftSpec
        >>> golden = SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),))
        >>> StateSpec(domain=golden).check().domain.alphabet.cardinality()
        2
    """

    neighborhood: object
    forbidden: Sequence[tuple[int, ...]]
    states: int = 2

    def _check(self) -> None:
        """Raise an error for forbidden patterns of different lengths or with values outside the states.

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
